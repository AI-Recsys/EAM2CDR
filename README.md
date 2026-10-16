# EAM²CDR

**Energy-guided Adaptive Multi-manifold Learning for Cross-Domain Recommendation**

EAM²CDR combines adaptive mixed-curvature encoding, density-constrained preference disentanglement, and selective cross-domain transfer. It learns from source and target user–item interactions and ranks items in the target domain.

The implementation is organized around two Python files:

- `main.py`: data loading, configuration, training, evaluation, and command-line entry point.
- `model.py`: geometric encoding, shared/specific preference modeling, transfer, and recommendation scores.

Experiment settings are in `config/`, and the included interaction files are in `dataset/`.

## Requirements

Install PyTorch for your CPU or CUDA environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The built-in demo uses PyTorch and `tqdm`. Dataset training also needs a local `RecBole-CDR-main` checkout under an ancestor of this project (for this workspace, `Task2/RecBole-CDR-main`).

## Quick Start

Run a small standalone example:

```bash
python main.py --demo --steps 5
python main.py --demo --geometry-mode product --steps 5
```

Train on the included Douban Movie → Book data with the short example configuration:

```bash
python main.py --config config/example.yaml --no-save
```

The demo checks optimization and target-domain scoring. The dataset command runs the RecBole-CDR training and evaluation pipeline.

## Main Options

```bash
python main.py --config config/douban-movie-book.yaml
python main.py --config config/example.yaml --warm-start /path/to/checkpoint.pth
python main.py --config config/example.yaml --checkpoint /path/to/checkpoint.pth
```

- `--config FILE [FILE ...]`: YAML configuration files; defaults to `config/example.yaml`.
- `--no-save`: train without saving a checkpoint.
- `--warm-start FILE`: initialize compatible model weights and continue training.
- `--checkpoint FILE`: evaluate a saved checkpoint without retraining.
- `--demo`: run the standalone example without RecBole-CDR.
- `--steps N`: optimization steps for the demo; defaults to `5`.
- `--geometry-mode {lorentz,product}`: geometry for the demo; defaults to `lorentz`.

Set dataset paths, training hyperparameters, and model switches in YAML. `config/douban-movie-book-how123-corrected-100epoch.yaml` provides a product-manifold configuration with the disentanglement and selective-transfer branches enabled; check its saved-data paths before using it.

## Model Components

- **Adaptive mixed-curvature encoding:** a product manifold combines spherical, Euclidean, and hyperbolic factors using topology and structural-energy gates. The `lorentz` backbone is also available.
- **Density-constrained disentanglement:** shared and domain-specific streams use manifold density energy, independence regularization, and multi-interest clustering.
- **Selective transfer:** confidence-weighted alignment acts on shared representations while the domain-specific branches are isolated from alignment gradients.
- **Recommendation objective:** pairwise ranking is trained with the enabled geometric, disentanglement, and transfer regularizers.

The YAML switches are `geometry_mode`, `enable_how2`, and `enable_how3`; HOW3 requires HOW2.

## Evaluation

Report **Precision, Recall, MRR, Hit Ratio, and NDCG** at **K = 5, 10, and 20**. Configure the selected YAML file as follows:

```yaml
metrics: [Recall, MRR, NDCG, Hit, Precision]
topk: [5, 10, 20]
valid_metric: NDCG@10
```

Example output format:

```text
@5:  Precision=..., Recall=..., MRR=..., Hit=..., NDCG=...
@10: Precision=..., Recall=..., MRR=..., Hit=..., NDCG=...
@20: Precision=..., Recall=..., MRR=..., Hit=..., NDCG=...
```

The included YAML files currently use smaller or single-K settings; the values above are required for the 5/10/20 evaluation. The model also provides user–item `score()` and target-domain `full_sort_target()` methods.
