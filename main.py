


from __future__ import annotations

import argparse
import math
import pickle
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Callable

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from model import HCTSEmCDR, collect_active_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = next(
    (parent / "RecBole-CDR-main" for parent in PROJECT_ROOT.parents
     if (parent / "RecBole-CDR-main").is_dir()),
    None,
)
if REFERENCE_ROOT is not None and str(REFERENCE_ROOT) not in sys.path:
    sys.path.append(str(REFERENCE_ROOT))

try:
    from recbole.utils import InputType, init_logger, init_seed, set_color
    from recbole_cdr.config import CDRConfig
    from recbole_cdr.data import create_dataset, data_preparation
    from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
    from recbole_cdr.trainer import CrossDomainTrainer
    _recbole_import_error = None
except ModuleNotFoundError as error:
    _recbole_import_error = str(error)
    class InputType:
        PAIRWISE = None
    CDRConfig = create_dataset = data_preparation = None
    CrossDomainRecommender = CrossDomainTrainer = object
    init_logger = init_seed = set_color = None


def _require_recbole() -> None:
    if REFERENCE_ROOT is None:
        raise RuntimeError("RecBole-CDR-main is required beside the Task2 project")
    if _recbole_import_error is not None:
        raise RuntimeError(
            "RecBole dependencies are missing; install requirements.txt. "
            + _recbole_import_error
        )


# RecBole data extraction and fixed splits

@dataclass(frozen=True)
class CrossDomainEdges:
    """Canonical source/target interaction edges extracted from RecBole data."""

    source_users: torch.Tensor
    source_items: torch.Tensor
    target_users: torch.Tensor
    target_items: torch.Tensor


def extract_cross_domain_edges(dataset) -> CrossDomainEdges:
    """Convert RecBole domain interaction features to plain tensors."""

    source = dataset.source_domain_dataset
    target = dataset.target_domain_dataset
    return CrossDomainEdges(
        source_users=torch.as_tensor(source.inter_feat[source.uid_field]),
        source_items=torch.as_tensor(source.inter_feat[source.iid_field]),
        target_users=torch.as_tensor(target.inter_feat[target.uid_field]),
        target_items=torch.as_tensor(target.inter_feat[target.iid_field]),
    )


def prepare_data_layer(
    config,
    dataset,
    *,
    project_root: str | Path,
    data_preparation_fn: Callable,
):
    """Prepare data or restore a compatible, trusted fixed split.

    Dependency injection keeps this input layer testable without importing the
    optional RecBole-CDR runtime at package import time.
    """

    configured_path = config["dataloaders_save_path"]
    if not configured_path:
        return data_preparation_fn(config, dataset)
    split_path = Path(configured_path)
    if not split_path.is_absolute():
        split_path = Path(project_root) / split_path
    if not split_path.is_file():
        return data_preparation_fn(config, dataset)

    with split_path.open("rb") as stream:
        dataloaders = pickle.load(stream)
    if not isinstance(dataloaders, tuple) or len(dataloaders) != 3:
        raise ValueError(f"invalid fixed dataloader tuple: {split_path}")

    saved_config = dataloaders[0].config
    compatibility_keys = ("seed", "repeatable", "eval_args", "data_path")
    mismatches = [
        key for key in compatibility_keys if config[key] != saved_config[key]
    ]
    for domain in ("source_domain", "target_domain"):
        current = config[domain]
        saved = saved_config[domain]
        for key in ("dataset", "USER_ID_FIELD", "ITEM_ID_FIELD", "RATING_FIELD"):
            if current[key] != saved[key]:
                mismatches.append(f"{domain}.{key}")
    if mismatches:
        raise ValueError(
            f"fixed dataloaders are incompatible at {mismatches}: {split_path}"
        )
    getLogger().info("Loading trusted fixed split dataloaders from: %s", split_path)
    return dataloaders


def collect_validation_overlap_users(
    model,
    valid_data,
    *,
    device,
    max_batches: int = 256,
) -> torch.Tensor:
    """Collect valid overlap-user IDs from a validation loader."""

    if max_batches <= 0:
        raise ValueError("max_batches must be positive")
    loader = valid_data[1] if isinstance(valid_data, tuple) else valid_data
    collected = []
    for batch_index, batch in enumerate(loader):
        interaction = batch[0] if isinstance(batch, (tuple, list)) else batch
        fields = (
            getattr(model, "TARGET_USER_ID", None),
            getattr(model, "SOURCE_USER_ID", None),
            getattr(model, "OVERLAP_ID", None),
        )
        for field in (value for value in fields if value is not None):
            try:
                values = interaction[field]
            except (KeyError, TypeError):
                continue
            values = values.reshape(-1)
            values = values[(values > 0) & (values < model.overlapped_num_users)]
            if values.numel() > 0:
                collected.append(values.to(device))
        if batch_index + 1 >= max_batches:
            break
    if not collected:
        return torch.arange(1, model.overlapped_num_users, device=device)
    return torch.unique(torch.cat(collected))


# Diagnostics formatting

def tensor_diagnostics_to_python(
    diagnostics: Mapping[str, Any], *, digits: int = 6
) -> dict[str, Any]:
    """Detach tensor diagnostics and convert them into log/JSON-safe values."""

    formatted: dict[str, Any] = {}
    for key, value in diagnostics.items():
        if torch.is_tensor(value):
            detached = value.detach().cpu()
            formatted[key] = (
                round(float(detached), digits)
                if detached.ndim == 0
                else [round(float(item), digits) for item in detached.reshape(-1)]
            )
        elif isinstance(value, float):
            formatted[key] = round(value, digits)
        else:
            formatted[key] = value
    return formatted


# Checkpoint loading and warm starts

def match_mutable_checkpoint_buffers(model, state_dict: dict) -> None:
    """Resize data-dependent buffers before strict checkpoint restoration."""

    key = "core.how3.calibrator.null_energy"
    stored = state_dict.get(key)
    core = getattr(model, "core", None)
    how3 = getattr(core, "how3", None)
    calibrator = getattr(how3, "calibrator", None)
    current = getattr(calibrator, "null_energy", None)
    if stored is None or current is None or current.shape == stored.shape:
        return
    calibrator.null_energy = current.new_empty(stored.shape)


def reset_interest_clusterers(model) -> list[str]:
    """Reset collapsed How2 centers while preserving other warm-start weights."""

    core = getattr(model, "core", None)
    reset_names = []
    for name in (
        "shared_clusterer",
        "source_private_clusterer",
        "target_private_clusterer",
    ):
        clusterer = getattr(core, name, None)
        if clusterer is None:
            continue
        clusterer.reset_parameters()
        reset_names.append(name)
    return reset_names


def validated_warm_start_score(checkpoint: dict, reset_names: list[str]):
    """Return the incumbent score only while its ranking state is unchanged."""

    score = checkpoint.get("best_valid_score")
    if score is None or reset_names or checkpoint.get("_shape_mismatch_names"):
        return None
    return float(score)


def load_trusted_checkpoint(
    trainer,
    checkpoint_file: str | Path,
    *,
    strict: bool = True,
    restore_schedule: bool = True,
    skip_shape_mismatches: bool = False,
) -> dict:
    """Load a locally generated RecBole checkpoint on PyTorch 2.6+."""

    checkpoint_path = Path(checkpoint_file).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    load_kwargs = {"map_location": trainer.device}
    try:
        checkpoint = torch.load(
            checkpoint_path,
            weights_only=False,
            **load_kwargs,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, **load_kwargs)

    state_dict = checkpoint["state_dict"]
    match_mutable_checkpoint_buffers(trainer.model, state_dict)
    skipped_shapes = []
    if skip_shape_mismatches:
        if strict:
            raise ValueError("skip_shape_mismatches requires strict=False")
        current_state = trainer.model.state_dict()
        compatible_state = {}
        for name, tensor in state_dict.items():
            current = current_state.get(name)
            if current is None or current.shape != tensor.shape:
                skipped_shapes.append(
                    (
                        name,
                        tuple(tensor.shape),
                        None if current is None else tuple(current.shape),
                    )
                )
                continue
            compatible_state[name] = tensor
        state_dict = compatible_state
    checkpoint["_shape_mismatch_names"] = [name for name, _, _ in skipped_shapes]
    incompatible = trainer.model.load_state_dict(state_dict, strict=strict)
    if restore_schedule:
        trainer.model.load_other_parameter(checkpoint.get("other_parameter"))
    if skipped_shapes:
        trainer.logger.info(
            "Warm-start skipped shape-mismatched tensors: %s", skipped_shapes
        )
    if not strict:
        trainer.logger.info(
            "Warm-start state: missing=%s unexpected=%s",
            list(incompatible.missing_keys),
            list(incompatible.unexpected_keys),
        )
    trainer.logger.info(
        "Loading trusted model structure and parameters from %s", checkpoint_path
    )
    return checkpoint


# RecBole model adapter

def _config_value(config, key, default):
    """Read an optional custom RecBole key with a stable fallback."""

    try:
        value = config[key]
    except (KeyError, TypeError):
        return default
    return default if value is None else value


class HCTS_EMCDR(CrossDomainRecommender):
    """Joint source/target HCTS model for RecBole-CDR."""

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        edges = extract_cross_domain_edges(dataset)
        source_curvature = config["source_curvature"]
        target_curvature = config["target_curvature"]
        source_curvature = source_curvature or config["source_domain"]["curvature"]
        target_curvature = target_curvature or config["target_domain"]["curvature"]
        max_transfer_users = _config_value(config, "max_transfer_users", None)
        self.core = HCTSEmCDR(
            total_num_users=self.total_num_users,
            total_num_items=self.total_num_items,
            target_num_users=self.target_num_users,
            target_num_items=self.target_num_items,
            overlapped_num_users=self.overlapped_num_users,
            overlapped_num_items=self.overlapped_num_items,
            source_edges=(edges.source_users, edges.source_items),
            target_edges=(edges.target_users, edges.target_items),
            embedding_size=int(config["embedding_size"]),
            num_layers=int(config["n_layers"]),
            source_curvature=float(source_curvature),
            target_curvature=float(target_curvature),
            learnable_curvature=bool(config["learnable_curvature"]),
            margin=float(config["margin"]),
            temperature=float(config["temperature"]),
            transfer_weight=float(config["transfer_weight"]),
            hierarchy_weight=float(config["hierarchy_weight"]),
            hierarchy_radius_sq=float(config["hierarchy_radius_sq"]),
            reg_weight=float(config["reg_weight"]),
            num_transfer_negatives=int(config["num_transfer_negatives"]),
            graph_dropout=float(config["graph_dropout"]),
            mapping_dropout=float(config["mapping_dropout"]),
            bidirectional=bool(config["bidirectional"]),
            enable_user_user=bool(config["u_u_cts"]),
            enable_user_item=bool(config["u_i_cts"]),
            enable_item_user=bool(config["i_u_cts"]),
            enable_item_item=bool(config["i_i_cts"]),
            detach_transferred=bool(config["detach_transferred"]),
            init_scale=float(config["init_scale"]),
            geometry_mode=str(_config_value(config, "geometry_mode", "lorentz")),
            manifold_dimensions=_config_value(config, "manifold_dimensions", None),
            source_spherical_curvature=float(
                _config_value(config, "source_spherical_curvature", 1.0)
            ),
            target_spherical_curvature=float(
                _config_value(config, "target_spherical_curvature", 1.0)
            ),
            source_hyperbolic_curvature=float(
                _config_value(config, "source_hyperbolic_curvature", source_curvature)
            ),
            target_hyperbolic_curvature=float(
                _config_value(config, "target_hyperbolic_curvature", target_curvature)
            ),
            gate_hidden_size=int(_config_value(config, "gate_hidden_size", 32)),
            gate_heads=int(_config_value(config, "gate_heads", 4)),
            gate_temperature=float(_config_value(config, "gate_temperature", 0.5)),
            topology_gate_strength=float(
                _config_value(config, "topology_gate_strength", 1.0)
            ),
            energy_gate_strength=float(
                _config_value(config, "energy_gate_strength", 1.0)
            ),
            gate_dropout=float(_config_value(config, "gate_dropout", 0.0)),
            energy_chunk_size=int(
                _config_value(config, "energy_chunk_size", 262144)
            ),
            contrastive_chunk_size=int(
                _config_value(config, "contrastive_chunk_size", 512)
            ),
            full_sort_item_chunk_size=int(
                _config_value(config, "full_sort_item_chunk_size", 4096)
            ),
            loss_type=str(_config_value(config, "loss_type", "margin")),
            normalize_energy_by_degree=bool(
                _config_value(config, "normalize_energy_by_degree", True)
            ),
            normalize_energy_per_node=bool(
                _config_value(config, "normalize_energy_per_node", True)
            ),
            detach_energy_feedback=bool(
                _config_value(config, "detach_energy_feedback", True)
            ),
            max_transfer_users=(
                None if max_transfer_users is None else int(max_transfer_users)
            ),
            normalize_component_distances=bool(
                _config_value(config, "normalize_component_distances", False)
            ),
            gate_warmup_epochs=int(
                _config_value(config, "gate_warmup_epochs", 0)
            ),
            gate_prior_epochs=int(
                _config_value(config, "gate_prior_epochs", 0)
            ),
            gate_ramp_epochs=_config_value(config, "gate_ramp_epochs", None),
            gate_initial_adaptive_scale=float(
                _config_value(config, "gate_initial_adaptive_scale", 0.0)
            ),
            equal_component_prior=bool(
                _config_value(config, "equal_component_prior", False)
            ),
            gate_shadow_train_during_prior=bool(
                _config_value(config, "gate_shadow_train_during_prior", False)
            ),
            gate_floor=float(_config_value(config, "gate_floor", 0.0)),
            gate_balance_weight=float(
                _config_value(config, "gate_balance_weight", 0.0)
            ),
            gate_node_kl_ratio=float(
                _config_value(config, "gate_node_kl_ratio", 0.0)
            ),
            enable_how2=bool(_config_value(config, "enable_how2", False)),
            how2_hidden_size=int(_config_value(config, "how2_hidden_size", 64)),
            disentanglement_weight=float(
                _config_value(config, "disentanglement_weight", 0.001)
            ),
            density_energy_weight=float(
                _config_value(
                    config, "density_energy_weight",
                    _config_value(config, "energy_weight", 0.001),
                )
            ),
            cluster_weight=float(
                _config_value(config, "cluster_weight", 0.001)
            ),
            density_energy_temperature=float(
                _config_value(config, "density_energy_temperature", 1.0)
            ),
            density_energy_margin=float(
                _config_value(config, "density_energy_margin", 1.0)
            ),
            density_energy_noise_scale=float(
                _config_value(config, "density_energy_noise_scale", 2.0)
            ),
            density_energy_l2=float(
                _config_value(config, "density_energy_l2", 0.001)
            ),
            num_preference_clusters=int(
                _config_value(config, "num_preference_clusters", 4)
            ),
            cluster_temperature=float(
                _config_value(config, "cluster_temperature", 0.2)
            ),
            cluster_balance_weight=float(
                _config_value(config, "cluster_balance_weight", 0.1)
            ),
            cluster_standardize_distances=bool(
                _config_value(config, "cluster_standardize_distances", False)
            ),
            cluster_distance_scale_floor=float(
                _config_value(config, "cluster_distance_scale_floor", 1.0)
            ),
            multi_interest_temperature=float(
                _config_value(config, "multi_interest_temperature", 0.2)
            ),
            interest_separation_weight=float(
                _config_value(config, "interest_separation_weight", 0.1)
            ),
            interest_separation_margin=float(
                _config_value(config, "interest_separation_margin", 1.0)
            ),
            w2_num_projections=int(
                _config_value(config, "w2_num_projections", 16)
            ),
            how2_warmup_epochs=int(
                _config_value(config, "how2_warmup_epochs", 0)
            ),
            how2_ramp_epochs=int(
                _config_value(config, "how2_ramp_epochs", 1)
            ),
            how2_residual_scale=float(
                _config_value(config, "how2_residual_scale", 1.0)
            ),
            how2_fixed_ramp_scale=_config_value(
                config, "how2_fixed_ramp_scale", None
            ),
            how2_initial_ramp_scale=float(
                _config_value(config, "how2_initial_ramp_scale", 0.0)
            ),
            detach_how2_auxiliary_base=bool(
                _config_value(config, "detach_how2_auxiliary_base", True)
            ),
            freeze_how1_after_warmup=bool(
                _config_value(config, "freeze_how1_after_warmup", False)
            ),
            enable_how3=bool(_config_value(config, "enable_how3", False)),
            selective_align_weight=float(
                _config_value(config, "selective_align_weight", 0.0)
            ),
            selective_contrastive_weight=float(
                _config_value(config, "selective_contrastive_weight", 0.0)
            ),
            align_loss_cap_ratio=float(
                _config_value(config, "align_loss_cap_ratio", 0.0)
            ),
            how3_start_epoch=int(_config_value(config, "how3_start_epoch", 0)),
            how3_ramp_epochs=int(_config_value(config, "how3_ramp_epochs", 0)),
            how3_initial_scale=float(
                _config_value(config, "how3_initial_scale", 0.0)
            ),
            energy_temperature=float(_config_value(config, "energy_temperature", 1.0)),
            energy_cross_weight=float(_config_value(config, "energy_cross_weight", 1.0)),
            energy_threshold_mode=str(_config_value(config, "energy_threshold_mode", "quantile")),
            energy_threshold_quantile=float(_config_value(config, "energy_threshold_quantile", 0.2)),
            energy_fdr_alpha=float(_config_value(config, "energy_fdr_alpha", 0.1)),
            energy_hard_screen=bool(_config_value(config, "energy_hard_screen", True)),
            energy_trust_max_fraction=float(_config_value(config, "energy_trust_max_fraction", 0.2)),
            align_detach_source=bool(_config_value(config, "align_detach_source", True)),
            shared_grad_mode=str(_config_value(config, "shared_grad_mode", "gradcraft+multibalance")),
            shared_grad_norm_ratio_max=float(_config_value(config, "shared_grad_norm_ratio_max", 0.5)),
            graph_encoder=str(_config_value(config, "graph_encoder", "hcts")),
        )
        self.phase = "BOTH"

    def other_parameter(self):
        """Save the non-tensor stage state together with the best checkpoint."""

        return {"training_schedule_state": self.core.training_schedule_state()}

    def load_other_parameter(self, para):
        """Restore the exact validation-time stage before testing a checkpoint."""

        if para:
            self.core.load_training_schedule_state(
                para.get("training_schedule_state")
            )

    def set_phase(self, phase):
        phase = str(phase).upper()
        if phase not in {"SOURCE", "TARGET", "BOTH", "OVERLAP"}:
            raise ValueError(f"unsupported training phase: {phase}")
        self.phase = phase

    def batch_overlap_users(self, interaction):
        users = torch.cat([
            interaction[self.SOURCE_USER_ID].reshape(-1),
            interaction[self.TARGET_USER_ID].reshape(-1),
        ])
        users = torch.unique(users)
        return users[(users > 0) & (users < self.overlapped_num_users)]

    @torch.no_grad()
    def calibrate_how3(self, validation_users):
        return self.core.fit_how3_threshold(validation_users)

    def _joint_breakdown(self, interaction):
        return self.core.loss(
            source_users=interaction[self.SOURCE_USER_ID],
            source_positive_items=interaction[self.SOURCE_ITEM_ID],
            source_negative_items=interaction[self.SOURCE_NEG_ITEM_ID],
            target_users=interaction[self.TARGET_USER_ID],
            target_positive_items=interaction[self.TARGET_ITEM_ID],
            target_negative_items=interaction[self.TARGET_NEG_ITEM_ID],
        )

    def how3_losses(self, interaction):
        return self.core.split_how123_loss(self._joint_breakdown(interaction))

    def calculate_loss(self, interaction):
        if self.phase == "SOURCE":
            return self.core.single_domain_loss(
                users=interaction[self.SOURCE_USER_ID],
                positive_items=interaction[self.SOURCE_ITEM_ID],
                negative_items=interaction[self.SOURCE_NEG_ITEM_ID],
                domain="source",
            )
        if self.phase == "TARGET":
            return self.core.single_domain_loss(
                users=interaction[self.TARGET_USER_ID],
                positive_items=interaction[self.TARGET_ITEM_ID],
                negative_items=interaction[self.TARGET_NEG_ITEM_ID],
                domain="target",
            )
        if self.phase == "OVERLAP":
            return self.core.overlap_loss(interaction[self.OVERLAP_ID])
        breakdown = self._joint_breakdown(interaction)
        losses = (
            breakdown.source,
            breakdown.target,
            self.core.transfer_weight * breakdown.transfer,
            self.core.hierarchy_weight * breakdown.hierarchy,
        )
        if self.core.gate_balance_weight > 0.0:
            losses = losses + (
                self.core.gate_balance_weight * breakdown.gate_balance,
            )
        if self.core.enable_how2:
            losses = losses + (
                self.core.disentanglement_weight * breakdown.disentanglement,
                self.core.density_energy_weight * breakdown.density_energy,
                self.core.cluster_weight * breakdown.clustering,
            )
        if self.core.enable_how3:
            losses = losses + (
                self.core.effective_selective_align_weight
                * breakdown.selective_align,
            )
        return losses

    def predict(self, interaction):
        return self.core.score(
            interaction[self.TARGET_USER_ID],
            interaction[self.TARGET_ITEM_ID],
            domain="target",
        )

    def full_sort_predict(self, interaction):
        return self.core.full_sort_target(interaction[self.TARGET_USER_ID])

    def get_visualization_data(self):
        return collect_active_embeddings(self.core)


# HOW1/2/3 training controls

def target_focused_primary_loss(losses, target_weight: float):
    """Reweight only the target ranking term during mature continuation."""

    target_weight = float(target_weight)
    if not math.isfinite(target_weight) or target_weight <= 0.0:
        raise ValueError("continuation_target_primary_weight must be finite and positive")
    return losses.primary + (target_weight - 1.0) * losses.breakdown.target


class How123CrossDomainTrainer(CrossDomainTrainer):
    """Split primary and How3 alignment gradients before the optimizer step."""

    def __init__(self, config, model):
        super().__init__(config, model)
        self._private_isolation_checked = False
        self.how3_diagnostics = {}
        self._warm_start_score = None
        self._checkpoint_saved_score = None
        custom_config = getattr(config, "final_config_dict", {})
        self.continuation_target_primary_weight = float(custom_config.get("continuation_target_primary_weight", 1.0))
        if not math.isfinite(self.continuation_target_primary_weight) or self.continuation_target_primary_weight <= 0.0:
            raise ValueError("continuation_target_primary_weight must be finite and positive")
        if self.continuation_target_primary_weight != 1.0:
            self.logger.info(
                "Target-focused continuation enabled: continuation_target_primary_weight=%s",
                self.continuation_target_primary_weight,
            )
        self._evaluation_overrides = {
            name: custom_config[key]
            for name, key in (
                ("normalize_component_distances", "eval_normalize_component_distances"),
                ("gate_adaptive_scale", "eval_gate_adaptive_scale"),
                ("gate_floor", "eval_gate_floor"),
                ("energy_gate_strength", "eval_energy_gate_strength"),
                ("how2_ramp_scale", "eval_how2_ramp_scale"),
                ("how2_residual_scale", "eval_how2_residual_scale"),
                ("multi_interest_temperature", "eval_multi_interest_temperature"),
            )
            if key in custom_config and custom_config[key] is not None
        }
        self.checkpoint_min_delta = float(
            custom_config.get("checkpoint_min_delta", 0.0)
        )
        if self.checkpoint_min_delta < 0.0:
            raise ValueError("checkpoint_min_delta must be non-negative")
        patience = int(custom_config.get("lr_plateau_patience", -1))
        factor = float(custom_config.get("lr_plateau_factor", 0.5))
        min_lr = float(custom_config.get("min_learning_rate", 0.0))
        threshold = float(custom_config.get("lr_plateau_threshold", 1e-4))
        self.rollback_on_lr_decay = bool(
            custom_config.get("rollback_on_lr_decay", False)
        )
        if patience < 0:
            self.validation_scheduler = None
        else:
            if not 0.0 < factor < 1.0:
                raise ValueError("lr_plateau_factor must be in (0, 1)")
            if min_lr < 0.0:
                raise ValueError("min_learning_rate must be non-negative")
            self.validation_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max" if self.valid_metric_bigger else "min",
                factor=factor,
                patience=patience,
                threshold=threshold,
                threshold_mode="abs",
                min_lr=min_lr,
            )
        if self._evaluation_overrides:
            self.logger.info(
                "Validation/test-only HOW controls enabled: %s",
                self._evaluation_overrides,
            )

    @contextmanager
    def evaluation_controls(self):
        """Apply validation-selected controls without changing training state."""

        if not self._evaluation_overrides:
            yield
            return
        core = self.model.core
        geometry = getattr(core, "product_geometry", None)
        gates = tuple(
            gate for gate in (
                getattr(core, "source_gate", None),
                getattr(core, "target_gate", None),
            ) if gate is not None
        )
        state = {
            "core_normalize": core.normalize_component_distances,
            "geometry_normalize": (
                None if geometry is None else geometry.normalize_component_distances
            ),
            "gate_adaptive_scale": tuple(gate.adaptive_scale for gate in gates),
            "gate_floor": tuple(gate.gate_floor for gate in gates),
            "energy_gate_strength": tuple(gate.energy_strength for gate in gates),
            "how2_ramp_scale": core._how2_ramp_scale,
            "how2_residual_scale": core.how2_residual_scale,
            "multi_interest_temperature": core.multi_interest_temperature,
        }
        try:
            if "normalize_component_distances" in self._evaluation_overrides:
                value = bool(self._evaluation_overrides["normalize_component_distances"])
                core.normalize_component_distances = value
                if geometry is not None:
                    geometry.normalize_component_distances = value
            if "gate_adaptive_scale" in self._evaluation_overrides:
                value = float(self._evaluation_overrides["gate_adaptive_scale"])
                for gate in gates:
                    gate.set_adaptive_scale(value)
            if "gate_floor" in self._evaluation_overrides:
                value = float(self._evaluation_overrides["gate_floor"])
                if not 0.0 <= value < 1.0:
                    raise ValueError("eval_gate_floor must be in [0, 1)")
                for gate in gates:
                    gate.gate_floor = value
            if "energy_gate_strength" in self._evaluation_overrides:
                value = float(self._evaluation_overrides["energy_gate_strength"])
                if value < 0.0:
                    raise ValueError("eval_energy_gate_strength must be non-negative")
                for gate in gates:
                    gate.energy_strength = value
            if "how2_ramp_scale" in self._evaluation_overrides:
                value = float(self._evaluation_overrides["how2_ramp_scale"])
                if not 0.0 <= value <= 1.0:
                    raise ValueError("eval_how2_ramp_scale must be in [0, 1]")
                core._how2_ramp_scale = value
            if "how2_residual_scale" in self._evaluation_overrides:
                value = float(self._evaluation_overrides["how2_residual_scale"])
                if not 0.0 <= value <= 1.0:
                    raise ValueError("eval_how2_residual_scale must be in [0, 1]")
                core.how2_residual_scale = value
            if "multi_interest_temperature" in self._evaluation_overrides:
                value = float(self._evaluation_overrides["multi_interest_temperature"])
                if value <= 0.0:
                    raise ValueError("eval_multi_interest_temperature must be positive")
                core.multi_interest_temperature = value
            core.clear_evaluation_cache()
            yield
        finally:
            core.normalize_component_distances = state["core_normalize"]
            if geometry is not None:
                geometry.normalize_component_distances = state["geometry_normalize"]
            for gate, value in zip(gates, state["gate_adaptive_scale"]):
                gate.set_adaptive_scale(value)
            for gate, value in zip(gates, state["gate_floor"]):
                gate.gate_floor = value
            for gate, value in zip(gates, state["energy_gate_strength"]):
                gate.energy_strength = value
            core._how2_ramp_scale = state["how2_ramp_scale"]
            core.how2_residual_scale = state["how2_residual_scale"]
            core.multi_interest_temperature = state["multi_interest_temperature"]
            core.clear_evaluation_cache()

    def _valid_epoch(self, valid_data, show_progress=False):
        with self.evaluation_controls():
            return super()._valid_epoch(valid_data, show_progress=show_progress)

    def seed_best_checkpoint(self, score):
        """Protect the validated warm-start model as the initial incumbent."""

        self._warm_start_score = None if score is None else float(score)
        if self._warm_start_score is not None and self.validation_scheduler is not None:
            # ReduceLROnPlateau otherwise treats the first degraded fine-tuning
            # score as its baseline and postpones the required rollback.
            self.validation_scheduler.best = self._warm_start_score

    def _step_validation_scheduler(self, valid_score):
        """Decay the learning rate only after validation has genuinely stalled."""

        if self.validation_scheduler is None:
            return
        previous = [group["lr"] for group in self.optimizer.param_groups]
        self.validation_scheduler.step(float(valid_score))
        current = [group["lr"] for group in self.optimizer.param_groups]
        if current != previous:
            self.logger.info(
                "Validation plateau: learning rate %s -> %s",
                previous,
                current,
            )
            if self.rollback_on_lr_decay:
                self._rollback_to_best_checkpoint()

    def _rollback_to_best_checkpoint(self):
        """Restart a lower-LR attempt from the protected validation incumbent."""

        load_kwargs = {"map_location": self.device}
        try:
            checkpoint = torch.load(
                self.saved_model_file, weights_only=False, **load_kwargs
            )
        except TypeError:
            checkpoint = torch.load(self.saved_model_file, **load_kwargs)
        state_dict = checkpoint["state_dict"]
        key = "core.how3.calibrator.null_energy"
        stored = state_dict.get(key)
        calibrator = getattr(getattr(self.model, "core", None), "how3", None)
        calibrator = getattr(calibrator, "calibrator", None)
        current = getattr(calibrator, "null_energy", None)
        if stored is not None and current is not None and current.shape != stored.shape:
            calibrator.null_energy = current.new_empty(stored.shape)
        self.model.load_state_dict(state_dict, strict=True)
        if hasattr(self.model, "load_other_parameter"):
            self.model.load_other_parameter(checkpoint.get("other_parameter"))
        self.optimizer.state.clear()
        self.logger.info(
            "Rolled back to validation incumbent after LR decay: %s",
            self.saved_model_file,
        )

    def fit(
        self,
        train_data,
        valid_data=None,
        verbose=True,
        saved=True,
        show_progress=False,
        callback_fn=None,
    ):
        """Compose LR scheduling with the runner's geometry callback."""

        def validation_callback(epoch_idx, valid_score):
            self._step_validation_scheduler(valid_score)
            if callback_fn is not None:
                callback_fn(epoch_idx, valid_score)

        return super().fit(
            train_data,
            valid_data=valid_data,
            verbose=verbose,
            saved=saved,
            show_progress=show_progress,
            callback_fn=validation_callback,
        )

    def _reinit(self, phase):
        super()._reinit(phase)
        self._checkpoint_saved_score = None
        if phase == 0 and self._warm_start_score is not None:
            self.best_valid_score = self._warm_start_score
            self._save_checkpoint(-1)
            self.logger.info(
                "Protected warm-start incumbent: valid_score=%.6f",
                self._warm_start_score,
            )

    def _save_checkpoint(self, epoch, verbose=True, **kwargs):
        """Keep rounded validation ties from replacing a stronger incumbent."""

        score = getattr(self, "best_valid_score", None)
        finite_score = score is not None and math.isfinite(float(score))
        if (
            epoch >= 0
            and self.checkpoint_min_delta > 0.0
            and finite_score
            and self._checkpoint_saved_score is not None
        ):
            if self.valid_metric_bigger:
                improvement = float(score) - self._checkpoint_saved_score
            else:
                improvement = self._checkpoint_saved_score - float(score)
            if improvement + 1e-12 < self.checkpoint_min_delta:
                return
        super()._save_checkpoint(epoch, verbose=verbose, **kwargs)
        if finite_score:
            self._checkpoint_saved_score = float(score)

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if loss_func is not None or self.model.phase != "BOTH":
            return super()._train_epoch(
                train_data, epoch_idx, loss_func=loss_func, show_progress=show_progress
            )

        self.model.train()
        core = self.model.core
        total_loss = None
        iter_data = (
            tqdm(train_data, total=len(train_data), ncols=100, desc=f"Train {epoch_idx:>5}")
            if show_progress
            else train_data
        )
        for interaction in iter_data:
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            losses = self.model.how3_losses(interaction)
            primary = target_focused_primary_loss(losses, self.continuation_target_primary_weight)
            align = losses.align
            self._check_nan(primary + align)

            use_control = (
                core.how3_active
                and core.how3.is_calibrated
                and core.selective_align_weight > 0
                and align.requires_grad
            )
            private = [p for p in core.private_parameters() if p.requires_grad]
            if use_control and not self._private_isolation_checked:
                overlap = self.model.batch_overlap_users(interaction)
                if overlap.numel() > 0:
                    core.assert_private_isolation(overlap.unique())
                    self._private_isolation_checked = True

            if not use_control:
                (primary + align).backward()
            else:
                names, shared = core.conflict_parameters()
                primary_grads = list(
                    torch.autograd.grad(primary, shared, retain_graph=True, allow_unused=True)
                )
                align_grads = list(
                    torch.autograd.grad(align, shared, retain_graph=True, allow_unused=True)
                )
                stats = core.how3.gradient_controller.apply(
                    shared, primary_grads, align_grads, names=names
                )
                if private:
                    for parameter, gradient in zip(
                        private,
                        torch.autograd.grad(primary, private, allow_unused=True),
                    ):
                        if gradient is not None:
                            parameter.grad = gradient.detach().clone()
                core.how3.diagnostics.gradient = stats.as_dict()
                core.how3.diagnostics.conflict_matrix = stats.per_group

            if self.clip_grad_norm:
                clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
            self.optimizer.step()

            breakdown = losses.breakdown
            components = (
                float(breakdown.source.detach()),
                float(breakdown.target.detach()),
                float((core.transfer_weight * breakdown.transfer).detach()),
                float((core.hierarchy_weight * breakdown.hierarchy).detach()),
                float((core.gate_balance_weight * breakdown.gate_balance).detach()),
                float((core.disentanglement_weight * breakdown.disentanglement).detach()),
                float((core.density_energy_weight * breakdown.density_energy).detach()),
                float((core.cluster_weight * breakdown.clustering).detach()),
                float(align.detach()),
            )
            total_loss = (
                components
                if total_loss is None
                else tuple(map(sum, zip(total_loss, components)))
            )

        self.how3_diagnostics = core.how3_diagnostics()
        self.logger.info("How123 diagnostics (epoch %s): %s", epoch_idx, self.how3_diagnostics)
        return total_loss


# Experiment pipeline

def progress_enabled(config) -> bool:
    return bool(config["show_progress"])


@contextmanager
def hide_runner_args_from_recbole():
    """Prevent RecBole from warning about args already handled by this runner."""
    original_argv = sys.argv[:]
    sys.argv = [original_argv[0]]
    try:
        yield
    finally:
        sys.argv = original_argv


def prepare_data(config, dataset):
    """Compatibility wrapper around the input layer's split preparation."""

    return prepare_data_layer(
        config,
        dataset,
        project_root=PROJECT_ROOT,
        data_preparation_fn=data_preparation,
    )


def run(
    config_files: list[str],
    saved: bool = True,
    checkpoint_file: str | None = None,
    warm_start_file: str | None = None,
    trainer_class_override=None,
) -> dict:
    _require_recbole()
    with hide_runner_args_from_recbole():
        config = CDRConfig(model=HCTS_EMCDR, config_file_list=config_files)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = prepare_data(config, dataset)

    init_seed(config["seed"], config["reproducibility"])
    model = HCTS_EMCDR(config, train_data.dataset).to(config["device"])
    logger.info(model)
    use_how3_trainer = bool(
        getattr(config, "final_config_dict", {}).get("how3_trainer", False)
    )
    how3_trainer_class = trainer_class_override or How123CrossDomainTrainer
    trainer_class = how3_trainer_class if use_how3_trainer else CrossDomainTrainer
    trainer = trainer_class(config, model)
    continuation_offset = int(
        getattr(config, "final_config_dict", {}).get("continuation_epoch_offset", 0)
    )
    configured_warm_start = getattr(config, "final_config_dict", {}).get(
        "warm_start_checkpoint"
    )
    warm_start_file = warm_start_file or configured_warm_start
    how3_recalibration_epochs = {
        int(epoch)
        for epoch in getattr(config, "final_config_dict", {}).get(
            "how3_recalibration_epochs", []
        )
    }

    def validation_overlap_users(max_batches=256):
        return collect_validation_overlap_users(
            model,
            valid_data,
            device=config["device"],
            max_batches=max_batches,
        )

    def log_geometry_diagnostics(epoch_idx, valid_score):
        if not model.core.uses_product_manifold:
            return
        global_epoch = continuation_offset + epoch_idx
        formatted = tensor_diagnostics_to_python(model.core.gate_diagnostics())
        logger.info(
            "geometry diagnostics epoch=%s valid_score=%s: %s",
            global_epoch,
            valid_score,
            formatted,
        )
        if (
            model.core.enable_how2
            and bool(
                getattr(config, "final_config_dict", {}).get(
                    "how2_restore_best_warmup", False
                )
            )
            and global_epoch == model.core.how2_warmup_epochs
        ):
            logger.info(
                "How2 handoff: restoring the best How1 warmup checkpoint before "
                "freezing and residual fine-tuning"
            )
            load_trusted_checkpoint(trainer, trainer.saved_model_file)
        next_epoch = global_epoch + 1
        should_calibrate_how3 = (
            model.core.enable_how3
            and next_epoch >= model.core.how3_start_epoch
            and (
                next_epoch in how3_recalibration_epochs
                or (
                    next_epoch == model.core.how3_start_epoch
                    and not model.core.how3.is_calibrated
                )
            )
        )
        if should_calibrate_how3:
            users = validation_overlap_users()
            threshold = model.calibrate_how3(users)
            logger.info(
                "How3 validation calibration: users=%s threshold=%.6f",
                users.numel(), threshold,
            )
        model.core.set_training_epoch(global_epoch + 1)

    if checkpoint_file is None:
        warm_checkpoint = None
        reset_names = []
        if warm_start_file:
            skip_shape_mismatches = bool(
                getattr(config, "final_config_dict", {}).get(
                    "warm_start_skip_shape_mismatches", False
                )
            )
            warm_checkpoint = load_trusted_checkpoint(
                trainer, warm_start_file, strict=False, restore_schedule=False,
                skip_shape_mismatches=skip_shape_mismatches,
            )
            if bool(
                getattr(config, "final_config_dict", {}).get(
                    "reset_interest_clusters_on_warm_start", False
                )
            ):
                reset_names = reset_interest_clusterers(model)
                logger.info(
                    "Reset warm-start How2 interest clusterers: %s",
                    reset_names,
                )
            if saved and hasattr(trainer, "seed_best_checkpoint"):
                incumbent_score = validated_warm_start_score(
                    warm_checkpoint, reset_names
                )
                score_override = getattr(config, "final_config_dict", {}).get(
                    "warm_start_valid_score_override"
                )
                shape_mismatches = warm_checkpoint.get("_shape_mismatch_names", [])
                if score_override is not None and shape_mismatches:
                    logger.info(
                        "Ignoring warm-start score override after architecture "
                        "migration with %s shape-mismatched tensors",
                        len(shape_mismatches),
                    )
                elif score_override is not None:
                    if reset_names:
                        raise ValueError(
                            "warm_start_valid_score_override cannot be used when "
                            "ranking parameters are reset"
                        )
                    incumbent_score = float(score_override)
                    if not math.isfinite(incumbent_score):
                        raise ValueError(
                            "warm_start_valid_score_override must be finite"
                        )
                    logger.info(
                        "Using externally validated warm-start score override: %.6f",
                        incumbent_score,
                    )
                if incumbent_score is None and reset_names:
                    logger.info(
                        "Skipped warm-start incumbent score because ranking "
                        "parameters were reset: %s",
                        reset_names,
                    )
                else:
                    trainer.seed_best_checkpoint(incumbent_score)
        model.core.set_training_epoch(continuation_offset)
        force_warm_start_calibration = bool(
            getattr(config, "final_config_dict", {}).get(
                "recalibrate_how3_on_warm_start", False
            )
        )
        if (
            model.core.enable_how3
            and model.core.how3_active
            and (
                not model.core.how3.is_calibrated
                or (warm_checkpoint is not None and force_warm_start_calibration)
            )
        ):
            users = validation_overlap_users()
            threshold = model.calibrate_how3(users)
            logger.info(
                "How3 pre-fit validation calibration: users=%s threshold=%.6f",
                users.numel(), threshold,
            )
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=saved,
            show_progress=progress_enabled(config),
            callback_fn=log_geometry_diagnostics,
        )
        if saved:
            load_trusted_checkpoint(trainer, trainer.saved_model_file)
    else:
        checkpoint = load_trusted_checkpoint(trainer, checkpoint_file)
        best_valid_score = checkpoint.get("best_valid_score")
        best_valid_result = None

    if hasattr(trainer, "evaluation_controls"):
        with trainer.evaluation_controls():
            test_result = trainer.evaluate(
                test_data,
                load_best_model=False,
                show_progress=progress_enabled(config),
            )
    else:
        test_result = trainer.evaluate(
            test_data,
            load_best_model=False,
            show_progress=progress_enabled(config),
        )
    if best_valid_result is not None:
        logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    elif best_valid_score is not None:
        logger.info(set_color("best valid score", "yellow") + f": {best_valid_score}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    return {
        "best_valid_score": best_valid_score,
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def run_demo(*, geometry_mode: str = "lorentz", steps: int = 5) -> None:
    """Run the existing small PyTorch example without RecBole."""

    if steps < 1:
        raise ValueError("steps must be positive")
    torch.manual_seed(2026)
    source_users = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 8, 8, 9, 9])
    source_items = torch.tensor([8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14])
    target_users = torch.tensor([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
    target_items = torch.tensor([1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7])
    model = HCTSEmCDR(
        total_num_users=11,
        total_num_items=15,
        target_num_users=8,
        target_num_items=8,
        overlapped_num_users=5,
        overlapped_num_items=1,
        source_edges=(source_users, source_items),
        target_edges=(target_users, target_items),
        embedding_size=8,
        num_layers=2,
        source_curvature=0.8,
        target_curvature=1.2,
        transfer_weight=0.05,
        hierarchy_weight=0.002,
        hierarchy_radius_sq=2.0,
        mapping_dropout=0.0,
        geometry_mode=geometry_mode,
    )
    batch = {
        "source_users": source_users,
        "source_positive_items": source_items,
        "source_negative_items": torch.tensor(
            [11, 12, 13, 14, 8, 9, 14, 8, 9, 10, 11, 12]
        ),
        "target_users": target_users,
        "target_positive_items": target_items,
        "target_negative_items": torch.tensor(
            [4, 5, 6, 7, 1, 2, 7, 1, 2, 3, 4, 5]
        ),
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    first = None
    for _ in range(steps):
        optimizer.zero_grad()
        loss = model.loss(**batch).total
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss: {loss}")
        first = float(loss.detach()) if first is None else first
        loss.backward()
        optimizer.step()
    model.eval()
    scores = model.full_sort_target(torch.tensor([1, 5])).reshape(
        2, model.target_num_items
    )
    print(f"total loss: {first:.6f} -> {float(loss.detach()):.6f}")
    print(
        "curvatures:",
        f"source={float(model.source_curvature.detach()):.4f}",
        f"target={float(model.target_curvature.detach()):.4f}",
    )
    print("target full-sort shape:", tuple(scores.shape))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        nargs="+",
        default=[str(PROJECT_ROOT / "config" / "example.yaml")],
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--demo", action="store_true", help="Run a small PyTorch example")
    parser.add_argument("--steps", type=int, default=5, help="Demo optimizer steps")
    parser.add_argument(
        "--geometry-mode", choices=("lorentz", "product"), default="lorentz",
        help="Geometry used by --demo",
    )
    parser.add_argument(
        "--warm-start",
        type=str,
        default=None,
        help="Warm-start compatible parameters and continue training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Skip training and evaluate a trusted project checkpoint.",
    )
    args = parser.parse_args()
    if args.demo:
        if args.steps < 1:
            parser.error("--steps must be positive")
        run_demo(geometry_mode=args.geometry_mode, steps=args.steps)
        return
    run(
        args.config,
        saved=not args.no_save,
        checkpoint_file=args.checkpoint,
        warm_start_file=args.warm_start,
    )


if __name__ == "__main__":
    main()
