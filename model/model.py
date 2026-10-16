

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# Interaction graph construction

def build_normalized_bipartite_adjacency(
    users: Tensor,
    items: Tensor,
    *,
    num_users: int,
    num_items: int,
) -> Tensor:
    """Build a symmetrically normalized sparse bipartite adjacency matrix."""

    if num_users < 0 or num_items < 0:
        raise ValueError("num_users and num_items must be non-negative")
    users = torch.as_tensor(users).long().reshape(-1).cpu()
    items = torch.as_tensor(items).long().reshape(-1).cpu()
    if users.numel() != items.numel():
        raise ValueError("users and items must contain the same number of edges")

    valid = (users >= 0) & (users < num_users) & (items >= 0) & (items < num_items)
    users = users[valid]
    items = items[valid] + num_users
    node_count = num_users + num_items
    if users.numel() == 0:
        indices = torch.empty((2, 0), dtype=torch.long)
        values = torch.empty(0, dtype=torch.float32)
        return torch.sparse_coo_tensor(
            indices, values, (node_count, node_count)
        ).coalesce()

    rows = torch.cat([users, items])
    columns = torch.cat([items, users])
    indices = torch.stack([rows, columns])
    values = torch.ones(rows.numel(), dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(
        indices, values, (node_count, node_count)
    ).coalesce()

    rows, columns = adjacency.indices()
    degree = torch.zeros(node_count, dtype=torch.float32)
    degree.scatter_add_(0, rows, adjacency.values())
    inv_sqrt_degree = degree.clamp_min(1.0).pow(-0.5)
    normalized_values = (
        adjacency.values() * inv_sqrt_degree[rows] * inv_sqrt_degree[columns]
    )
    return torch.sparse_coo_tensor(
        adjacency.indices(), normalized_values, adjacency.size()
    ).coalesce()


def build_neighbor_csr(
    users: Tensor,
    items: Tensor,
    *,
    num_users: int,
) -> tuple[Tensor, Tensor]:
    """Build CSR offsets and item values for vectorized neighbor sampling."""

    if num_users < 0:
        raise ValueError("num_users must be non-negative")
    users = torch.as_tensor(users).long().reshape(-1).cpu()
    items = torch.as_tensor(items).long().reshape(-1).cpu()
    if users.numel() != items.numel():
        raise ValueError("users and items must contain the same number of edges")

    valid = (users >= 0) & (users < num_users)
    users, items = users[valid], items[valid]
    order = torch.argsort(users, stable=True)
    sorted_users, sorted_items = users[order], items[order]
    counts = torch.bincount(sorted_users, minlength=num_users)
    offsets = torch.zeros(num_users + 1, dtype=torch.long)
    offsets[1:] = counts.cumsum(0)
    return offsets, sorted_items


# Lorentz geometry

def _curvature(c: float | Tensor, reference: Tensor) -> Tensor:
    value = torch.as_tensor(c, dtype=reference.dtype, device=reference.device)
    if torch.any(value <= 0):
        raise ValueError("curvature magnitude c must be positive")
    return value


def arcosh(x: Tensor) -> Tensor:
    """Numerically stable inverse hyperbolic cosine."""

    return torch.log(x + torch.sqrt((x.square() - 1.0).clamp_min(0.0)))


def minkowski_dot(x: Tensor, y: Tensor, keepdim: bool = False) -> Tensor:
    """Lorentzian inner product with signature (-,+,...,+)."""

    if x.size(-1) != y.size(-1):
        raise ValueError("Lorentz vectors must have the same ambient dimension")
    result = -(x[..., 0] * y[..., 0]) + (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    return result.unsqueeze(-1) if keepdim else result


def project(x: Tensor, c: float | Tensor) -> Tensor:
    """Project ambient vectors onto the upper sheet <x,x>_L=-1/c."""

    curvature = _curvature(c, x)
    radius_sq = curvature.reciprocal()
    spatial = x[..., 1:]
    time = torch.sqrt((radius_sq + spatial.square().sum(dim=-1, keepdim=True)).clamp_min(0.0))
    return torch.cat([time, spatial], dim=-1)


def expmap0(v: Tensor, c: float | Tensor, max_theta: float = 15.0) -> Tensor:
    """Map Euclidean spatial tangent vectors to the Lorentz hyperboloid.

    The input has D spatial coordinates and the output has D+1 Lorentz
    coordinates. This convention avoids the unused first embedding coordinate
    in the original HCTS implementation.
    """

    curvature = _curvature(c, v)
    radius = curvature.rsqrt()
    norm = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    theta = (norm / radius).clamp(max=max_theta)
    time = radius * torch.cosh(theta)
    scale = torch.where(
        norm > 0,
        radius * torch.sinh(theta) / norm.clamp_min(torch.finfo(v.dtype).tiny),
        torch.ones_like(norm),
    )
    return torch.cat([time, scale * v], dim=-1)


def logmap0(x: Tensor, c: float | Tensor) -> Tensor:
    """Map Lorentz points to D-dimensional spatial tangent coordinates."""

    curvature = _curvature(c, x)
    x = project(x, curvature)
    radius = curvature.rsqrt()
    spatial = x[..., 1:]
    spatial_norm = torch.linalg.vector_norm(spatial, dim=-1, keepdim=True)
    eps = 1e-7 if x.dtype == torch.float32 else 1e-15
    theta = arcosh((x[..., :1] / radius).clamp_min(1.0 + eps))
    scale = torch.where(
        spatial_norm > 0,
        radius * theta / spatial_norm.clamp_min(torch.finfo(x.dtype).tiny),
        torch.zeros_like(spatial_norm),
    )
    return scale * spatial


def squared_distance(x: Tensor, y: Tensor, c: float | Tensor) -> Tensor:
    """Squared geodesic distance between broadcast-compatible Lorentz points."""

    curvature = _curvature(c, x)
    eps = 1e-6 if x.dtype == torch.float32 else 1e-12
    theta = (-curvature * minkowski_dot(project(x, curvature), project(y, curvature))).clamp_min(
        1.0 + eps
    )
    return arcosh(theta).square() / curvature


def pairwise_squared_distance(x: Tensor, y: Tensor, c: float | Tensor) -> Tensor:
    """Pairwise squared distances for x[B,D+1] and y[N,D+1]."""

    if x.ndim != 2 or y.ndim != 2 or x.size(-1) != y.size(-1):
        raise ValueError("x and y must have shapes [B,D+1] and [N,D+1]")
    curvature = _curvature(c, x)
    x = project(x, curvature)
    y = project(y, curvature)
    product = -(x[:, :1] @ y[:, :1].transpose(0, 1))
    product = product + x[:, 1:] @ y[:, 1:].transpose(0, 1)
    eps = 1e-6 if x.dtype == torch.float32 else 1e-12
    theta = (-curvature * product).clamp_min(1.0 + eps)
    return arcosh(theta).square() / curvature


def manifold_error(x: Tensor, c: float | Tensor) -> Tensor:
    """Absolute violation of the hyperboloid constraint."""

    curvature = _curvature(c, x)
    return (minkowski_dot(x, x) + curvature.reciprocal()).abs()


class PositiveCurvature(nn.Module):
    """Positive fixed or learnable curvature magnitude."""

    def __init__(self, value: float, learnable: bool = True, minimum: float = 1e-4) -> None:
        super().__init__()
        if value <= minimum:
            raise ValueError(f"initial curvature must be greater than {minimum}")
        self.minimum = float(minimum)
        raw = math.log(math.expm1(value - minimum))
        if learnable:
            self.raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))
        else:
            self.register_buffer("raw", torch.tensor(raw, dtype=torch.float32))

    def forward(self) -> Tensor:
        return F.softplus(self.raw) + self.minimum


# Graph propagation and tangent maps

class LorentzLinear(nn.Module):
    """Linear map between origin tangent spaces of two hyperboloids."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.dropout = float(dropout)
        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain=math.sqrt(2.0))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor, c_in: Tensor, c_out: Tensor) -> Tensor:
        tangent = logmap0(x, c_in)
        weight = F.dropout(self.weight, p=self.dropout, training=self.training)
        mapped = F.linear(tangent, weight, self.bias)
        return project(expmap0(mapped, c_out), c_out)


class HCTSGraphEncoder(nn.Module):
    """Tangent-space graph propagation used by the HCTS backbone."""

    def __init__(self, num_layers: int, dropout: float = 0.0) -> None:
        super().__init__()
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

    def forward(self, tangent: Tensor, normalized_adjacency: Tensor) -> Tensor:
        states = [tangent]
        hidden = tangent
        for _ in range(self.num_layers):
            hidden = torch.sparse.mm(normalized_adjacency, hidden)
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
            states.append(hidden)
        return torch.stack(states, dim=0).mean(dim=0)


# Product manifold geometry and gates

def _safe_norm(x: Tensor) -> Tensor:
    return torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(
        torch.finfo(x.dtype).tiny
    )


def _artanh(x: Tensor) -> Tensor:
    eps = 1e-6 if x.dtype == torch.float32 else 1e-12
    x = x.clamp(min=-1.0 + eps, max=1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def split_component_dimensions(total_dim: int, dimensions: Sequence[int] | None) -> tuple[int, int, int]:
    """Return intrinsic dimensions in (sphere, Euclidean, hyperbolic) order."""

    if dimensions is None:
        base, remainder = divmod(int(total_dim), 3)
        result = [base, base, base]
        for index in range(remainder):
            result[index] += 1
    else:
        result = [int(value) for value in dimensions]
    if len(result) != 3 or min(result) < 1 or sum(result) != int(total_dim):
        raise ValueError(
            "manifold_dimensions must contain three positive values whose sum "
            f"equals embedding_size={total_dim}; received {result}"
        )
    return result[0], result[1], result[2]


class ProductManifoldGeometry:
    """Numerically stable geometry for S^d_s x E^d_e x H^d_h.

    The sphere is represented extrinsically, so a product point has one more
    coordinate than its origin-tangent representation.
    """

    component_names = ("sphere", "euclidean", "hyperbolic")

    def __init__(
        self,
        dimensions: Sequence[int],
        *,
        normalize_component_distances: bool = False,
    ) -> None:
        self.dimensions = tuple(int(value) for value in dimensions)
        if len(self.dimensions) != 3 or min(self.dimensions) < 1:
            raise ValueError("product manifold requires three positive dimensions")
        sphere_dim, euclidean_dim, hyperbolic_dim = self.dimensions
        self.point_dimensions = (sphere_dim + 1, euclidean_dim, hyperbolic_dim)
        self.tangent_dim = sum(self.dimensions)
        self.point_dim = sum(self.point_dimensions)
        self.normalize_component_distances = bool(normalize_component_distances)

    def component_metric_scale(self, reference: Tensor) -> Tensor:
        """Return dimension/local-metric scales for comparable factor distances."""

        sphere_dim, euclidean_dim, hyperbolic_dim = self.dimensions
        return reference.new_tensor(
            [sphere_dim, euclidean_dim, 4.0 * hyperbolic_dim]
        )

    def calibrate_component_distances(self, distances: Tensor) -> Tensor:
        """Put spherical, Euclidean and Poincare squared distances on one scale."""

        if distances.size(-1) != 3:
            raise ValueError("component distances must have a final dimension of 3")
        return distances / self.component_metric_scale(distances)

    @staticmethod
    def _positive_curvature(curvature: Tensor | float, reference: Tensor) -> Tensor:
        value = torch.as_tensor(curvature, dtype=reference.dtype, device=reference.device)
        return value.clamp_min(torch.finfo(reference.dtype).eps)

    def split_tangent(self, tangent: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return torch.split(tangent, self.dimensions, dim=-1)

    def split_point(self, point: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return torch.split(point, self.point_dimensions, dim=-1)

    def expmap0(
        self,
        tangent: Tensor,
        spherical_curvature: Tensor | float,
        hyperbolic_curvature: Tensor | float,
    ) -> Tensor:
        sphere_tangent, euclidean, hyperbolic_tangent = self.split_tangent(tangent)

        sphere_c = self._positive_curvature(spherical_curvature, tangent)
        sphere_radius = sphere_c.rsqrt()
        sphere_norm = _safe_norm(sphere_tangent)
        sphere_angle = (sphere_norm / sphere_radius).clamp(max=math.pi - 1e-5)
        sphere_scale = sphere_radius * torch.sin(sphere_angle) / sphere_norm
        sphere = torch.cat(
            [sphere_radius * torch.cos(sphere_angle), sphere_scale * sphere_tangent],
            dim=-1,
        )

        hyper_c = self._positive_curvature(hyperbolic_curvature, tangent)
        sqrt_hyper_c = hyper_c.sqrt()
        hyper_norm = _safe_norm(hyperbolic_tangent)
        hyper_scale = torch.tanh((sqrt_hyper_c * hyper_norm).clamp(max=15.0))
        hyper_scale = hyper_scale / (sqrt_hyper_c * hyper_norm)
        hyperbolic = self.project_hyperbolic(hyper_scale * hyperbolic_tangent, hyper_c)
        return torch.cat([sphere, euclidean, hyperbolic], dim=-1)

    def logmap0(
        self,
        point: Tensor,
        spherical_curvature: Tensor | float,
        hyperbolic_curvature: Tensor | float,
    ) -> Tensor:
        sphere, euclidean, hyperbolic = self.split_point(point)

        sphere_c = self._positive_curvature(spherical_curvature, point)
        sphere_radius = sphere_c.rsqrt()
        sphere = self.project_sphere(sphere, sphere_c)
        spatial = sphere[..., 1:]
        spatial_norm = _safe_norm(spatial)
        angle = torch.atan2(spatial_norm, sphere[..., :1])
        sphere_tangent = sphere_radius * angle * spatial / spatial_norm

        hyper_c = self._positive_curvature(hyperbolic_curvature, point)
        hyperbolic = self.project_hyperbolic(hyperbolic, hyper_c)
        hyper_norm = _safe_norm(hyperbolic)
        hyper_tangent = _artanh(hyper_c.sqrt() * hyper_norm)
        hyper_tangent = hyper_tangent * hyperbolic / (hyper_c.sqrt() * hyper_norm)
        return torch.cat([sphere_tangent, euclidean, hyper_tangent], dim=-1)

    def project_sphere(self, point: Tensor, curvature: Tensor | float) -> Tensor:
        value = self._positive_curvature(curvature, point)
        return value.rsqrt() * point / _safe_norm(point)

    def project_hyperbolic(self, point: Tensor, curvature: Tensor | float) -> Tensor:
        value = self._positive_curvature(curvature, point)
        max_norm = (1.0 - (1e-5 if point.dtype == torch.float32 else 1e-10)) * value.rsqrt()
        norm = _safe_norm(point)
        return point * (max_norm / norm).clamp(max=1.0)

    def component_squared_distances(
        self,
        x: Tensor,
        y: Tensor,
        spherical_curvature: Tensor | float,
        hyperbolic_curvature: Tensor | float,
    ) -> Tensor:
        """Broadcast-compatible component distances without vector Mobius tensors."""

        sphere_x, euclidean_x, hyperbolic_x = self.split_point(x)
        sphere_y, euclidean_y, hyperbolic_y = self.split_point(y)
        tiny = torch.finfo(x.dtype).tiny

        sphere_c = self._positive_curvature(spherical_curvature, x)
        sphere_radius = sphere_c.rsqrt()
        sphere_x = self.project_sphere(sphere_x, sphere_c)
        sphere_y = self.project_sphere(sphere_y, sphere_c)
        sphere_x_sq = sphere_x.square().sum(dim=-1)
        sphere_y_sq = sphere_y.square().sum(dim=-1)
        sphere_xy = (sphere_x * sphere_y).sum(dim=-1)
        chord_sq = (sphere_x_sq + sphere_y_sq - 2.0 * sphere_xy).clamp_min(tiny)
        eps = 1e-6 if x.dtype == torch.float32 else 1e-12
        sine_half_angle = chord_sq.sqrt() / (2.0 * sphere_radius)
        spherical_distance = 2.0 * sphere_radius * torch.asin(
            sine_half_angle.clamp(max=1.0 - eps)
        )

        euclidean_x_sq = euclidean_x.square().sum(dim=-1)
        euclidean_y_sq = euclidean_y.square().sum(dim=-1)
        euclidean_xy = (euclidean_x * euclidean_y).sum(dim=-1)
        euclidean_distance_sq = (
            euclidean_x_sq + euclidean_y_sq - 2.0 * euclidean_xy
        ).clamp_min(0.0)

        hyper_c = self._positive_curvature(hyperbolic_curvature, x)
        sqrt_hyper_c = hyper_c.sqrt()
        hyperbolic_x = self.project_hyperbolic(hyperbolic_x, hyper_c)
        hyperbolic_y = self.project_hyperbolic(hyperbolic_y, hyper_c)
        hyperbolic_x_sq = hyperbolic_x.square().sum(dim=-1)
        hyperbolic_y_sq = hyperbolic_y.square().sum(dim=-1)
        hyperbolic_xy = (hyperbolic_x * hyperbolic_y).sum(dim=-1)
        difference_sq = (
            hyperbolic_x_sq + hyperbolic_y_sq - 2.0 * hyperbolic_xy
        ).clamp_min(tiny)
        conformal_denominator = (
            (1.0 - hyper_c * hyperbolic_x_sq).clamp_min(tiny)
            * (1.0 - hyper_c * hyperbolic_y_sq).clamp_min(tiny)
        )
        hyperbolic_distance = 2.0 * torch.asinh(
            (hyper_c * difference_sq / conformal_denominator).clamp_min(tiny).sqrt()
        ) / sqrt_hyper_c

        return torch.stack(
            [spherical_distance.square(), euclidean_distance_sq, hyperbolic_distance.square()],
            dim=-1,
        )

    def squared_distance(
        self,
        x: Tensor,
        y: Tensor,
        weights: Tensor,
        spherical_curvature: Tensor | float,
        hyperbolic_curvature: Tensor | float,
    ) -> Tensor:
        components = self.component_squared_distances(
            x, y, spherical_curvature, hyperbolic_curvature
        )
        if self.normalize_component_distances:
            components = self.calibrate_component_distances(components)
        return (weights * components).sum(dim=-1)

    def pairwise_squared_distance(
        self,
        x: Tensor,
        y: Tensor,
        x_weights: Tensor,
        y_weights: Tensor,
        spherical_curvature: Tensor | float,
        hyperbolic_curvature: Tensor | float,
    ) -> Tensor:
        """Memory-efficient pairwise product distance for two point matrices."""

        if x.ndim != 2 or y.ndim != 2:
            raise ValueError("pairwise distance expects x and y to be matrices")
        sphere_x, euclidean_x, hyperbolic_x = self.split_point(x)
        sphere_y, euclidean_y, hyperbolic_y = self.split_point(y)
        tiny = torch.finfo(x.dtype).tiny

        sphere_c = self._positive_curvature(spherical_curvature, x)
        sphere_radius = sphere_c.rsqrt()
        sphere_x = self.project_sphere(sphere_x, sphere_c)
        sphere_y = self.project_sphere(sphere_y, sphere_c)
        chord_sq = (
            sphere_x.square().sum(-1, keepdim=True)
            + sphere_y.square().sum(-1).unsqueeze(0)
            - 2.0 * (sphere_x @ sphere_y.transpose(0, 1))
        ).clamp_min(tiny)
        eps = 1e-6 if x.dtype == torch.float32 else 1e-12
        sphere_distance_sq = (
            2.0
            * sphere_radius
            * torch.asin(
                (chord_sq.sqrt() / (2.0 * sphere_radius)).clamp(max=1.0 - eps)
            )
        ).square()

        euclidean_distance_sq = (
            euclidean_x.square().sum(-1, keepdim=True)
            + euclidean_y.square().sum(-1).unsqueeze(0)
            - 2.0 * (euclidean_x @ euclidean_y.transpose(0, 1))
        ).clamp_min(0.0)

        hyper_c = self._positive_curvature(hyperbolic_curvature, x)
        hyperbolic_x = self.project_hyperbolic(hyperbolic_x, hyper_c)
        hyperbolic_y = self.project_hyperbolic(hyperbolic_y, hyper_c)
        hyperbolic_x_sq = hyperbolic_x.square().sum(-1, keepdim=True)
        hyperbolic_y_sq = hyperbolic_y.square().sum(-1).unsqueeze(0)
        difference_sq = (
            hyperbolic_x_sq
            + hyperbolic_y_sq
            - 2.0 * (hyperbolic_x @ hyperbolic_y.transpose(0, 1))
        ).clamp_min(tiny)
        denominator = (
            (1.0 - hyper_c * hyperbolic_x_sq).clamp_min(tiny)
            * (1.0 - hyper_c * hyperbolic_y_sq).clamp_min(tiny)
        )
        hyperbolic_distance_sq = (
            2.0
            * torch.asinh(
                (hyper_c * difference_sq / denominator).clamp_min(tiny).sqrt()
            )
            / hyper_c.sqrt()
        ).square()

        sphere_weights = 0.5 * (x_weights[:, :1] + y_weights[:, 0].unsqueeze(0))
        euclidean_weights = 0.5 * (x_weights[:, 1:2] + y_weights[:, 1].unsqueeze(0))
        hyperbolic_weights = 0.5 * (x_weights[:, 2:3] + y_weights[:, 2].unsqueeze(0))
        components = torch.stack(
            [sphere_distance_sq, euclidean_distance_sq, hyperbolic_distance_sq],
            dim=-1,
        )
        if self.normalize_component_distances:
            components = self.calibrate_component_distances(components)
        weights = torch.stack(
            [sphere_weights, euclidean_weights, hyperbolic_weights], dim=-1
        )
        return (weights * components).sum(dim=-1)

    def structural_energy(
        self,
        points: Tensor,
        edge_index: Tensor,
        spherical_curvature: Tensor | float,
        hyperbolic_curvature: Tensor | float,
        chunk_size: int = 262144,
    ) -> Tensor:
        """Compute local component energies in edge chunks to bound peak memory."""

        if chunk_size < 1:
            raise ValueError("energy chunk_size must be positive")
        source, target = edge_index
        energy = points.new_zeros((points.size(0), 3))
        for start in range(0, source.numel(), chunk_size):
            stop = min(start + chunk_size, source.numel())
            source_chunk = source[start:stop]
            target_chunk = target[start:stop]
            edge_energy = self.component_squared_distances(
                points[source_chunk],
                points[target_chunk],
                spherical_curvature,
                hyperbolic_curvature,
            )
            energy.index_add_(0, source_chunk, edge_energy)
        return energy

    def calibrate_structural_energy(
        self,
        energy: Tensor,
        degree: Tensor,
        *,
        normalize_by_degree: bool = True,
        normalize_per_node: bool = True,
    ) -> Tensor:
        """Put component energies on comparable local metric and dimension scales.

        Poincare distances are locally twice the coordinate-space norm, so their
        squared energy carries a factor four in addition to component dimension.
        """

        if energy.size(-1) != 3 or degree.shape != energy.shape[:-1]:
            raise ValueError("energy must be [N,3] and degree must be [N]")
        calibrated = self.calibrate_component_distances(energy)
        if normalize_by_degree:
            calibrated = calibrated / degree.clamp_min(1.0).unsqueeze(-1)
        if normalize_per_node:
            reference = calibrated.mean(dim=-1, keepdim=True)
            calibrated = calibrated / reference.clamp_min(
                torch.finfo(energy.dtype).eps
            )
        return calibrated


def build_topology_features(adjacency: Tensor) -> Tensor:
    """Degree and bipartite two-hop closure features for every graph node.

    Ordinary triangle clustering is identically zero on a user-item bipartite
    graph.  The diagonal of A_norm^2 is its useful bipartite counterpart: the
    probability of returning to the node after two normalized graph steps.
    """

    adjacency = adjacency.coalesce()
    rows, columns = adjacency.indices()
    node_count = adjacency.size(0)
    degree = adjacency.values().new_zeros(node_count)
    degree.index_add_(0, rows, torch.ones_like(adjacency.values()))
    log_degree = torch.log1p(degree)
    if log_degree.numel() > 0:
        log_degree = log_degree / log_degree.max().clamp_min(1.0)
    closure = adjacency.values().new_zeros(node_count)
    closure.index_add_(0, rows, adjacency.values().square())
    return torch.stack([log_degree, closure], dim=-1)


class TopologyEnergyGate(nn.Module):
    """GCN + multi-head topology prior refined by manifold energy feedback."""

    def __init__(
        self,
        hidden_size: int = 32,
        heads: int = 4,
        temperature: float = 0.5,
        topology_strength: float = 1.0,
        energy_strength: float = 1.0,
        dropout: float = 0.0,
        component_prior: Sequence[float] | None = None,
        gate_floor: float = 0.0,
        shadow_train_during_prior: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or heads < 1 or hidden_size % heads != 0:
            raise ValueError("gate_hidden_size must be positive and divisible by gate_heads")
        if temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        if topology_strength < 0 or energy_strength < 0:
            raise ValueError("gate strengths must be non-negative")
        if not 0.0 <= gate_floor < 1.0:
            raise ValueError("gate_floor must be in [0, 1)")
        prior = torch.tensor(
            [1.0 / 3.0] * 3 if component_prior is None else list(component_prior),
            dtype=torch.float32,
        )
        if prior.numel() != 3 or not torch.isfinite(prior).all() or (prior <= 0).any():
            raise ValueError("component_prior must contain three positive finite values")
        prior = prior / prior.sum()
        self.temperature = float(temperature)
        self.topology_strength = float(topology_strength)
        self.energy_strength = float(energy_strength)
        self.gate_floor = float(gate_floor)
        self.shadow_train_during_prior = bool(shadow_train_during_prior)
        self.adaptive_scale = 1.0
        self.latest_learned_gates: Tensor | None = None
        self.register_buffer("component_prior", prior, persistent=False)
        self.input_projection = nn.Linear(2, hidden_size)
        self.gcn_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.component_tokens = nn.Parameter(torch.empty(3, hidden_size))
        self.attention = nn.MultiheadAttention(
            hidden_size, heads, dropout=dropout, batch_first=True
        )
        self.output = nn.Linear(hidden_size, 1, bias=False)
        self.component_bias = nn.Parameter(torch.zeros(3))
        self.normalization = nn.LayerNorm(hidden_size)
        self.token_normalization = nn.LayerNorm(hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.component_tokens)
        nn.init.zeros_(self.component_bias)

    def set_adaptive_scale(self, scale: float) -> None:
        """Blend from a stable dimension prior (0) to the learned gate (1)."""

        if not 0.0 <= scale <= 1.0:
            raise ValueError("adaptive gate scale must be in [0, 1]")
        self.adaptive_scale = float(scale)

    def forward(
        self,
        topology_features: Tensor,
        adjacency: Tensor,
        structural_energy: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden = F.gelu(self.input_projection(topology_features))
        neighborhood = torch.sparse.mm(adjacency, hidden)
        hidden = self.normalization(hidden + self.gcn_projection(neighborhood))
        tokens = hidden.unsqueeze(1) + self.component_tokens.unsqueeze(0)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        attended = self.token_normalization(tokens + attended)
        topology_logits = self.output(attended).squeeze(-1) + self.component_bias
        topology_prior = F.softmax(topology_logits, dim=-1)

        energy_logits = -structural_energy / self.temperature
        combined_logits = self.topology_strength * torch.log(topology_prior.clamp_min(1e-12))
        combined_logits = combined_logits + self.energy_strength * energy_logits
        learned_gates = F.softmax(combined_logits, dim=-1)
        self.latest_learned_gates = learned_gates
        stable_prior = self.component_prior.to(
            dtype=learned_gates.dtype, device=learned_gates.device
        ).expand_as(learned_gates)
        gates = (
            (1.0 - self.adaptive_scale) * stable_prior
            + self.adaptive_scale * learned_gates
        )
        if (
            self.training
            and self.shadow_train_during_prior
            and self.adaptive_scale == 0.0
        ):
            # Keep exact prior values while preparing the learned gate for handoff.
            gates = gates + learned_gates - learned_gates.detach()
        if self.gate_floor > 0.0:
            gates = (1.0 - self.gate_floor) * gates + self.gate_floor * stable_prior
        return gates, topology_prior, structural_energy


class ProductManifoldLinear(nn.Module):
    """Factor-preserving transfer between product-manifold tangent spaces."""

    def __init__(self, geometry: ProductManifoldGeometry, dropout: float = 0.0) -> None:
        super().__init__()
        self.geometry = geometry
        self.dropout = float(dropout)
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(size, size)) for size in geometry.dimensions]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(size)) for size in geometry.dimensions]
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight, bias in zip(self.weights, self.biases):
            nn.init.xavier_uniform_(weight, gain=math.sqrt(2.0))
            nn.init.zeros_(bias)

    def forward(
        self,
        point: Tensor,
        source_spherical_curvature: Tensor,
        source_hyperbolic_curvature: Tensor,
        target_spherical_curvature: Tensor,
        target_hyperbolic_curvature: Tensor,
    ) -> Tensor:
        tangent = self.geometry.logmap0(
            point, source_spherical_curvature, source_hyperbolic_curvature
        )
        tangent_parts = self.geometry.split_tangent(tangent)
        mapped_parts = []
        for part, weight, bias in zip(tangent_parts, self.weights, self.biases):
            dropped_weight = F.dropout(weight, p=self.dropout, training=self.training)
            mapped_parts.append(F.linear(part, dropped_weight, bias))
        mapped = torch.cat(mapped_parts, dim=-1)
        return self.geometry.expmap0(
            mapped, target_spherical_curvature, target_hyperbolic_curvature
        )


# Density energy and preference decomposition

@dataclass
class DualStreamOutput:
    """Shared, private and fused views of one domain's user preferences."""

    fused: Tensor
    shared: Tensor
    private: Tensor
    shared_tangent: Tensor
    private_tangent: Tensor
    shared_auxiliary_tangent: Tensor
    private_auxiliary_tangent: Tensor


class ResidualTangentBranch(nn.Module):
    """Near-identity residual MLP operating in one manifold factor."""

    def __init__(self, dimension: int, hidden_size: int) -> None:
        super().__init__()
        if dimension <= 0 or hidden_size <= 0:
            raise ValueError("dimension and hidden_size must be positive")
        self.normalization = nn.LayerNorm(dimension)
        self.input_layer = nn.Linear(dimension, hidden_size)
        self.output_layer = nn.Linear(hidden_size, dimension)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def residual(self, tangent: Tensor) -> Tensor:
        return self.output_layer(
            F.silu(self.input_layer(self.normalization(tangent)))
        )

    def forward(self, tangent: Tensor) -> Tensor:
        return tangent + self.residual(tangent)


class FactorizedTangentBranch(nn.Module):
    """Independent residual branches for S/E/H factors (or one Lorentz factor)."""

    def __init__(self, dimensions: Sequence[int], hidden_size: int) -> None:
        super().__init__()
        self.dimensions = tuple(int(value) for value in dimensions)
        if not self.dimensions or min(self.dimensions) <= 0:
            raise ValueError("component dimensions must be positive")
        self.branches = nn.ModuleList(
            ResidualTangentBranch(dimension, hidden_size)
            for dimension in self.dimensions
        )

    def reset_parameters(self) -> None:
        for branch in self.branches:
            branch.reset_parameters()

    def residual(self, tangent: Tensor) -> Tensor:
        if tangent.size(-1) != sum(self.dimensions):
            raise ValueError("tangent dimension does not match component dimensions")
        components = torch.split(tangent, self.dimensions, dim=-1)
        return torch.cat(
            [
                branch.residual(value)
                for branch, value in zip(self.branches, components)
            ],
            dim=-1,
        )

    def forward(self, tangent: Tensor) -> Tensor:
        return tangent + self.residual(tangent)


class SharedPrivateDisentangler(nn.Module):
    """Factor-wise shared branch plus one private branch for each domain."""

    def __init__(
        self,
        dimension: int,
        hidden_size: int,
        component_dimensions: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        dimensions = (int(dimension),) if component_dimensions is None else tuple(
            int(value) for value in component_dimensions
        )
        if sum(dimensions) != int(dimension):
            raise ValueError("component dimensions must sum to dimension")
        self.shared_branch = FactorizedTangentBranch(dimensions, hidden_size)
        self.private_branches = nn.ModuleDict(
            {
                "source": FactorizedTangentBranch(dimensions, hidden_size),
                "target": FactorizedTangentBranch(dimensions, hidden_size),
            }
        )

    def from_tangent(
        self,
        tangent: Tensor,
        expmap: Callable[[Tensor], Tensor],
        domain: str,
        *,
        residual_scale: float = 1.0,
        detach_auxiliary_base: bool = False,
    ) -> DualStreamOutput:
        if domain not in self.private_branches:
            raise ValueError("domain must be source or target")
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        branch_input = tangent.detach() if detach_auxiliary_base else tangent
        shared_residual = self.shared_branch.residual(branch_input)
        private_residual = self.private_branches[domain].residual(branch_input)
        shared_tangent = tangent + residual_scale * shared_residual
        private_tangent = residual_scale * private_residual
        fused_tangent = tangent + 0.5 * residual_scale * (
            shared_residual + private_residual
        )
        auxiliary_base = tangent.detach() if detach_auxiliary_base else tangent
        shared_auxiliary_tangent = (
            auxiliary_base + residual_scale * shared_residual
        )
        private_auxiliary_tangent = residual_scale * private_residual
        return DualStreamOutput(
            fused=expmap(fused_tangent),
            shared=expmap(shared_tangent),
            private=expmap(private_tangent),
            shared_tangent=shared_tangent,
            private_tangent=private_tangent,
            shared_auxiliary_tangent=shared_auxiliary_tangent,
            private_auxiliary_tangent=private_auxiliary_tangent,
        )

    def forward(self, points: Tensor, curvature: Tensor, domain: str) -> DualStreamOutput:
        """Backward-compatible Lorentz entry point used by single-manifold tests."""

        tangent = logmap0(points, curvature)
        return self.from_tangent(
            tangent,
            lambda value: expmap0(value, curvature),
            domain,
        )


def riemannian_log_volume(
    tangent: Tensor,
    component_dimensions: Sequence[int],
    *,
    spherical_curvature: Tensor | float | None = None,
    hyperbolic_curvature: Tensor | float | None = None,
) -> Tensor:
    dimensions = tuple(int(value) for value in component_dimensions)
    if sum(dimensions) != tangent.size(-1) or min(dimensions) <= 0:
        raise ValueError("component dimensions must match the tangent dimension")
    eps = torch.finfo(tangent.dtype).eps
    factors = torch.split(tangent, dimensions, dim=-1)
    result = tangent.new_zeros(tangent.shape[:-1])
    if len(factors) == 3 and spherical_curvature is not None:
        curvature = torch.as_tensor(spherical_curvature, dtype=tangent.dtype, device=tangent.device).clamp_min(eps)
        radius = factors[0].norm(dim=-1)
        scaled = (curvature.sqrt() * radius).clamp(max=math.pi - 1e-4)
        ratio = torch.sin(scaled).clamp_min(eps) / scaled.clamp_min(eps)
        result = result + (dimensions[0] - 1) * torch.log(ratio)
    hyperbolic_index = 2 if len(factors) == 3 else 0
    if hyperbolic_curvature is not None:
        curvature = torch.as_tensor(hyperbolic_curvature, dtype=tangent.dtype, device=tangent.device).clamp_min(eps)
        radius = factors[hyperbolic_index].norm(dim=-1)
        scaled = (curvature.sqrt() * radius).clamp(max=20.0)
        ratio = torch.sinh(scaled).clamp_min(eps) / scaled.clamp_min(eps)
        result = result + (dimensions[hyperbolic_index] - 1) * torch.log(ratio)
    return result


class ManifoldDensityEnergy(nn.Module):
    """Dimension- and volume-normalized density energy in normal coordinates."""

    def __init__(
        self,
        dimension: int,
        hidden_size: int,
        component_dimensions: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if dimension <= 0 or hidden_size <= 0:
            raise ValueError("dimension and hidden_size must be positive")
        self.dimension = int(dimension)
        self.component_dimensions = (
            (self.dimension,)
            if component_dimensions is None
            else tuple(int(value) for value in component_dimensions)
        )
        if sum(self.component_dimensions) != self.dimension:
            raise ValueError("component dimensions must sum to dimension")
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def from_tangent(
        self,
        tangent: Tensor,
        *,
        spherical_curvature: Tensor | float | None = None,
        hyperbolic_curvature: Tensor | float | None = None,
    ) -> Tensor:
        factors = torch.split(tangent, self.component_dimensions, dim=-1)
        normalized = torch.cat(
            [value / math.sqrt(dimension) for value, dimension in zip(factors, self.component_dimensions)],
            dim=-1,
        )
        coordinate_energy = self.network(normalized).squeeze(-1)
        log_volume = riemannian_log_volume(
            tangent,
            self.component_dimensions,
            spherical_curvature=spherical_curvature,
            hyperbolic_curvature=hyperbolic_curvature,
        )
        return (coordinate_energy + log_volume) / self.dimension

    def forward(self, points: Tensor, curvature: Tensor) -> Tensor:
        return self.from_tangent(
            logmap0(points, curvature), hyperbolic_curvature=curvature
        )

    def contrastive_loss(
        self,
        positive_tangent: Tensor,
        *,
        temperature: float,
        margin: float,
        noise_scale: float,
        energy_l2: float,
        spherical_curvature: Tensor | float | None = None,
        hyperbolic_curvature: Tensor | float | None = None,
    ) -> Tensor:
        """Positive/negative phase plus an energy gap prevents constant collapse."""

        if positive_tangent.numel() == 0:
            return sum(parameter.sum() * 0.0 for parameter in self.parameters())
        if temperature <= 0 or noise_scale <= 0:
            raise ValueError("temperature and noise_scale must be positive")
        # Match the positive norm distribution so the EBM cannot solve the
        # task through a trivial norm threshold. Random directions provide
        # angular hard negatives in the same tangent-space radial shell.
        eps = torch.finfo(positive_tangent.dtype).eps
        negative_direction = F.normalize(
            torch.randn_like(positive_tangent), dim=-1, eps=eps
        )
        positive_norm = positive_tangent.detach().norm(
            dim=-1, keepdim=True
        )
        radial_jitter = (
            1.0 + 0.1 * noise_scale * torch.randn_like(positive_norm)
        ).clamp(0.5, 1.5)
        negative_tangent = negative_direction * positive_norm * radial_jitter
        energy_arguments = {
            "spherical_curvature": spherical_curvature,
            "hyperbolic_curvature": hyperbolic_curvature,
        }
        positive_energy = self.from_tangent(positive_tangent, **energy_arguments)
        negative_energy = self.from_tangent(negative_tangent, **energy_arguments)
        nce = F.softplus(positive_energy / temperature).mean()
        nce = nce + F.softplus(-negative_energy / temperature).mean()
        gap = F.relu(margin + positive_energy - negative_energy).mean()
        regularization = (
            positive_energy.square().mean() + negative_energy.square().mean()
        )
        return nce + gap + energy_l2 * regularization


class SlicedWassersteinIndependence(nn.Module):
    """Analytic projected Gaussian W2 between joint and product marginals."""

    def __init__(self, dimension: int, num_projections: int = 16) -> None:
        super().__init__()
        if dimension <= 0 or num_projections <= 0:
            raise ValueError("dimension and num_projections must be positive")
        row = torch.arange(1, num_projections + 1, dtype=torch.float32).unsqueeze(1)
        column = torch.arange(1, 2 * dimension + 1, dtype=torch.float32).unsqueeze(0)
        directions = torch.sin(1.61803398875 * row * column)
        directions = directions + torch.cos(0.70710678118 * row * (column + 0.5))
        self.register_buffer("directions", F.normalize(directions, dim=-1))

    def forward(self, shared: Tensor, private: Tensor) -> Tensor:
        if shared.shape != private.shape or shared.ndim != 2:
            raise ValueError("shared and private must have the same [B,D] shape")
        if shared.size(0) < 2:
            return (shared.sum() + private.sum()) * 0.0

        eps = torch.finfo(shared.dtype).eps
        shared_standard = shared - shared.mean(dim=0, keepdim=True)
        private_standard = private - private.mean(dim=0, keepdim=True)
        shared_standard = shared_standard / torch.sqrt(
            shared_standard.square().mean(dim=0, keepdim=True) + eps
        )
        private_standard = private_standard / torch.sqrt(
            private_standard.square().mean(dim=0, keepdim=True) + eps
        )
        joint = torch.cat([shared_standard, private_standard], dim=-1)
        product = torch.cat(
            [shared_standard, private_standard.roll(1, dims=0)], dim=-1
        )
        directions = self.directions.to(dtype=joint.dtype)
        joint_projection = joint @ directions.transpose(0, 1)
        product_projection = product @ directions.transpose(0, 1)
        mean_distance = (
            joint_projection.mean(0) - product_projection.mean(0)
        ).square()
        joint_std = torch.sqrt(joint_projection.var(0, unbiased=False) + eps)
        product_std = torch.sqrt(product_projection.var(0, unbiased=False) + eps)
        return (mean_distance + (joint_std - product_std).square()).mean()


class EntropyManifoldClusterer(nn.Module):
    """Soft K-Means objective accepting Lorentz or external product distances."""

    def __init__(
        self,
        dimension: int,
        num_clusters: int,
        temperature: float,
        balance_weight: float,
        standardize_distances: bool = False,
        distance_scale_floor: float = 1.0,
    ) -> None:
        super().__init__()
        if dimension <= 0 or num_clusters < 2:
            raise ValueError(
                "dimension must be positive and num_clusters must be at least two"
            )
        if temperature <= 0 or balance_weight < 0 or distance_scale_floor <= 0:
            raise ValueError(
                "temperature and distance_scale_floor must be positive and "
                "balance_weight non-negative"
            )
        self.dimension = int(dimension)
        self.num_clusters = int(num_clusters)
        self.temperature = float(temperature)
        self.balance_weight = float(balance_weight)
        self.standardize_distances = bool(standardize_distances)
        self.distance_scale_floor = float(distance_scale_floor)
        self.centers_tangent = nn.Parameter(torch.empty(num_clusters, dimension))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.centers_tangent, std=0.05)

    def assignments_from_distances(self, distances: Tensor) -> Tensor:
        if distances.ndim != 2 or distances.size(1) != self.num_clusters:
            raise ValueError("distances must have shape [B,num_clusters]")
        assignment_distances = distances
        if self.standardize_distances:
            assignment_distances = distances - distances.mean(
                dim=-1, keepdim=True
            )
            # Treat the row scale as calibration, not as a trainable shortcut.
            # The floor prevents tiny distance differences from being amplified.
            distance_scale = assignment_distances.std(
                dim=-1, unbiased=False, keepdim=True
            ).detach().clamp_min(self.distance_scale_floor)
            assignment_distances = assignment_distances / distance_scale
        return F.softmax(-assignment_distances / self.temperature, dim=-1)

    def loss_from_distances(self, distances: Tensor) -> tuple[Tensor, Tensor]:
        if distances.size(0) == 0:
            zero = self.centers_tangent.sum() * 0.0
            return zero, distances.new_empty((0, self.num_clusters))
        assignments = self.assignments_from_distances(distances)
        eps = torch.finfo(assignments.dtype).eps
        log_clusters = math.log(self.num_clusters)
        sample_entropy = -(
            assignments * assignments.clamp_min(eps).log()
        ).sum(dim=-1).mean() / log_clusters
        marginal = assignments.mean(dim=0)
        marginal_entropy = -(
            marginal * marginal.clamp_min(eps).log()
        ).sum() / log_clusters
        balance_penalty = 1.0 - marginal_entropy
        compactness = (assignments * distances).sum(dim=-1).mean() / self.dimension
        loss = compactness + sample_entropy + self.balance_weight * balance_penalty
        return loss, assignments

    def assignments(self, points: Tensor, curvature: Tensor) -> tuple[Tensor, Tensor]:
        centers = expmap0(self.centers_tangent, curvature)
        distances = pairwise_squared_distance(points, centers, curvature)
        return self.assignments_from_distances(distances), distances

    def separation_loss(self, margin: float) -> Tensor:
        if margin <= 0:
            return self.centers_tangent.sum() * 0.0
        distances = torch.cdist(
            self.centers_tangent, self.centers_tangent
        ).square()
        mask = ~torch.eye(
            self.num_clusters,
            dtype=torch.bool,
            device=distances.device,
        )
        return F.relu(float(margin) - distances[mask]).mean()

    def forward(self, points: Tensor, curvature: Tensor) -> tuple[Tensor, Tensor]:
        if points.size(0) == 0:
            zero = self.centers_tangent.sum() * 0.0
            return zero, points.new_empty((0, self.num_clusters))
        _, distances = self.assignments(points, curvature)
        return self.loss_from_distances(distances)


def symmetrized_assignment_kl(first: Tensor, second: Tensor) -> Tensor:
    """Symmetric KL between paired soft cluster assignments."""

    if first.shape != second.shape:
        raise ValueError("assignment tensors must have the same shape")
    if first.numel() == 0:
        return (first.sum() + second.sum()) * 0.0
    eps = torch.finfo(first.dtype).eps
    first = first.clamp_min(eps)
    second = second.clamp_min(eps)
    first_to_second = (first * (first.log() - second.log())).sum(dim=-1)
    second_to_first = (second * (second.log() - first.log())).sum(dim=-1)
    return 0.5 * (first_to_second + second_to_first).mean()


# Transfer energy and calibration

def knn_mean_squared_distance(
    points: Tensor,
    reference_points: Tensor,
    curvature: Tensor,
    *,
    topk: int,
    reference_chunk_size: int = 2048,
) -> Tensor:
    """Mean kNN distance without materializing the full point-reference matrix."""

    if points.size(0) == 0:
        return points.new_zeros(0)
    if reference_points.size(0) == 0:
        return points.new_zeros(points.size(0))
    k = max(1, min(int(topk), reference_points.size(0)))
    best: Tensor | None = None
    for start in range(0, reference_points.size(0), int(reference_chunk_size)):
        chunk = reference_points[start : start + int(reference_chunk_size)]
        distances = pairwise_squared_distance(points, chunk, curvature)
        chunk_best = distances.topk(min(k, chunk.size(0)), largest=False).values
        candidates = chunk_best if best is None else torch.cat([best, chunk_best], dim=1)
        best = candidates.topk(min(k, candidates.size(1)), largest=False).values
    if best is None:
        return points.new_zeros(points.size(0))
    return best.mean(dim=1)


class RunningStandardizer(nn.Module):
    """EMA mean/std standardizer keeping two energy terms commensurable."""

    def __init__(self, momentum: float = 0.9) -> None:
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.momentum = float(momentum)
        self.register_buffer("mean", torch.tensor(float("nan")))
        self.register_buffer("var", torch.tensor(float("nan")))

    @property
    def is_fitted(self) -> bool:
        return bool(
            torch.isfinite(self.mean).item() and torch.isfinite(self.var).item()
        )

    @torch.no_grad()
    def fit(self, values: Tensor) -> None:
        """Fit one global scale from an aggregated calibration sample."""

        values = values.detach().reshape(-1)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return
        self.mean.copy_(values.mean().to(self.mean))
        self.var.copy_(values.var(unbiased=False).to(self.var))

    def transform(self, values: Tensor) -> Tensor:
        """Apply fitted statistics without updating them."""

        if values.numel() == 0:
            return values
        if not self.is_fitted:
            batch_mean = values.detach().mean()
            batch_var = values.detach().var(unbiased=False)
            return (values - batch_mean) / batch_var.clamp_min(1e-8).sqrt()
        return (values - self.mean) / self.var.clamp_min(1e-8).sqrt()

    def forward(self, values: Tensor) -> Tensor:
        if values.numel() == 0:
            return values
        detached = values.detach()
        batch_mean = detached.mean()
        batch_var = detached.var(unbiased=False)
        if self.training:
            with torch.no_grad():
                if bool(torch.isnan(self.mean).item()):
                    self.mean.copy_(batch_mean)
                    self.var.copy_(batch_var)
                else:
                    momentum = self.momentum
                    self.mean.mul_(momentum).add_(batch_mean * (1.0 - momentum))
                    self.var.mul_(momentum).add_(batch_var * (1.0 - momentum))
        return self.transform(values)


def density_energy(
    points: Tensor,
    reference_points: Tensor,
    curvature: Tensor,
    *,
    topk: int = 8,
    density_weight: float = 1.0,
    radius_weight: float = 0.05,
) -> Tensor:
    """Manifold density energy ``E_den``: kNN geodesic energy plus radial deviation."""

    if points.size(0) == 0:
        return points.new_zeros(0)
    if reference_points.size(0) == 0:
        return points.new_zeros(points.size(0))

    reference_points = reference_points.detach()
    neighbor_energy = knn_mean_squared_distance(
        points,
        reference_points,
        curvature,
        topk=topk,
    )
    energy = density_weight * neighbor_energy
    if radius_weight > 0:
        point_radius = logmap0(points, curvature).square().sum(dim=-1)
        ref_radius = logmap0(reference_points, curvature).square().sum(dim=-1)
        scale = ref_radius.std(unbiased=False).clamp_min(1e-6)
        energy = energy + radius_weight * (point_radius - ref_radius.mean()).abs() / scale
    return energy


def cross_consistency_energy(
    *,
    source_users: Tensor,
    source_items: Tensor,
    mapped_users: Tensor,
    mapped_items: Tensor,
    roundtrip_users: Tensor | None,
    reference_points: Tensor,
    destination_users: Tensor | None = None,
    destination_items: Tensor | None = None,
    c_from: Tensor,
    c_to: Tensor,
    topk: int = 8,
) -> Tensor:
    """Cross-domain consistency cost ``D_cross``.

    The three terms answer "is this preference expressible in the destination
    domain?" without reusing the alignment objective itself, which would make the
    gate self-confirming:

    1. round-trip consistency of the bidirectional mapper;
    2. user-item relational consistency across the two curvatures;
    3. consistency with the same overlapping user's destination representation;
    4. destination-side neighbourhood support of the mapped point.
    """

    if mapped_users.size(0) == 0:
        return mapped_users.new_zeros(0)

    terms: list[Tensor] = []
    if roundtrip_users is not None:
        terms.append(squared_distance(roundtrip_users, source_users, c_from))

    source_relation = squared_distance(source_users, source_items, c_from)
    mapped_relation = squared_distance(mapped_users, mapped_items, c_to)
    terms.append((source_relation - mapped_relation).abs())

    if destination_users is not None:
        terms.append(squared_distance(mapped_users, destination_users, c_to))
    if destination_items is not None:
        terms.append(squared_distance(mapped_items, destination_items, c_to))

    if reference_points.size(0) > 0:
        support = knn_mean_squared_distance(
            mapped_users,
            reference_points.detach(),
            c_to,
            topk=topk,
        )
        terms.append(support)
    return torch.stack(terms, dim=0).mean(dim=0)


def trust_weights(energy: Tensor, temperature: float) -> Tensor:
    """Energy trust weights ``w_trust = softmax(-E_tr / tau_E)``."""

    if energy.numel() == 0:
        return energy
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    detached = energy.detach()
    stabilized = detached - detached.min()
    return F.softmax(-stabilized / temperature, dim=0)


def weighted_wasserstein2(weights: Tensor, sample_w2: Tensor) -> Tensor:
    """Energy weighted empirical ``W_2^2`` under the given coupling."""

    return (weights * sample_w2).sum()


def _weighted_quantiles(values: Tensor, weights: Tensor, quantiles: Tensor) -> Tensor:
    positive = weights.detach() > 0
    if not bool(positive.any()):
        positive = torch.ones_like(positive)
    values, weights = values[positive], weights[positive]
    order = torch.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order].clamp_min(0.0)
    sorted_weights = sorted_weights / sorted_weights.sum().clamp_min(1e-12)
    centers = sorted_weights.cumsum(0) - 0.5 * sorted_weights
    centers = torch.cat([centers.new_zeros(1), centers, centers.new_ones(1)])
    padded_values = torch.cat([sorted_values[:1], sorted_values, sorted_values[-1:]])
    upper = torch.searchsorted(centers.detach(), quantiles, right=True)
    upper = upper.clamp(1, centers.numel() - 1)
    lower = upper - 1
    fraction = (quantiles - centers[lower]) / (
        centers[upper] - centers[lower]
    ).clamp_min(1e-12)
    return padded_values[lower] + fraction * (padded_values[upper] - padded_values[lower])


def weighted_sliced_wasserstein2(
    source: Tensor,
    destination: Tensor,
    source_weights: Tensor,
    destination_weights: Tensor,
    *,
    num_projections: int = 16,
    num_quantiles: int = 32,
) -> Tensor:
    if source.ndim != 2 or destination.ndim != 2 or source.size(1) != destination.size(1):
        raise ValueError("source and destination must be [N,D] and [M,D]")
    if source.size(0) == 0 or destination.size(0) == 0:
        return (source.sum() + destination.sum()) * 0.0
    if source_weights.numel() != source.size(0) or destination_weights.numel() != destination.size(0):
        raise ValueError("distribution weights must match the number of atoms")
    if num_projections <= 0 or num_quantiles <= 1:
        raise ValueError("num_projections and num_quantiles must be positive")
    row = torch.arange(1, num_projections + 1, dtype=source.dtype, device=source.device).unsqueeze(1)
    column = torch.arange(1, source.size(1) + 1, dtype=source.dtype, device=source.device).unsqueeze(0)
    directions = torch.sin(1.61803398875 * row * column)
    directions = directions + torch.cos(0.70710678118 * row * (column + 0.5))
    directions = F.normalize(directions, dim=-1)
    source_projection = source @ directions.transpose(0, 1)
    destination_projection = destination @ directions.transpose(0, 1)
    quantiles = torch.linspace(0.0, 1.0, num_quantiles, dtype=source.dtype, device=source.device)
    distances = []
    for projection in range(num_projections):
        source_quantiles = _weighted_quantiles(
            source_projection[:, projection], source_weights, quantiles
        )
        destination_quantiles = _weighted_quantiles(
            destination_projection[:, projection], destination_weights, quantiles
        )
        distances.append((source_quantiles - destination_quantiles).square().mean())
    return torch.stack(distances).mean()


def weighted_energy_distance(
    source: Tensor,
    destination: Tensor,
    source_weights: Tensor,
    destination_weights: Tensor,
) -> Tensor:
    source_weights = source_weights / source_weights.sum().clamp_min(1e-12)
    destination_weights = destination_weights / destination_weights.sum().clamp_min(1e-12)
    cross = (
        torch.cdist(source, destination)
        * source_weights[:, None]
        * destination_weights[None, :]
    ).sum()
    within_source = (
        torch.cdist(source, source)
        * source_weights[:, None]
        * source_weights[None, :]
    ).sum()
    within_destination = (
        torch.cdist(destination, destination)
        * destination_weights[:, None]
        * destination_weights[None, :]
    ).sum()
    return (2.0 * cross - within_source - within_destination).clamp_min(0.0)


def weighted_cramer_energy(
    weights: Tensor,
    source_points: Tensor,
    destination_points: Tensor,
    curvature: Tensor,
) -> Tensor:
    """Weighted Cramer / energy distance in the tangent space.

    ``ED = 2 E||x-y|| - E||x-x'|| - E||y-y'||`` avoids the high-dimensional
    gradient bias of Wasserstein alignment (section 5.2.7).
    """

    if source_points.size(0) == 0:
        return source_points.sum() * 0.0
    x = logmap0(source_points, curvature)
    y = logmap0(destination_points, curvature)
    weights = weights.reshape(-1, 1)
    cross = (torch.cdist(x, y) * weights).sum(dim=0).mean()
    within_x = (torch.cdist(x, x) * weights).sum(dim=0).mean()
    within_y = torch.cdist(y, y).mean()
    return (2.0 * cross - within_x - within_y).clamp_min(0.0)


@dataclass
class PseudoLabelReport:
    threshold: float
    selected: int
    precision: float
    recall: float
    pollution: float
    accepted: bool


class EnergyThresholdCalibrator(nn.Module):
    """Validation-set energy threshold via quantile or Benjamini-Hochberg FDR.

    A hand-picked cross-dataset constant is explicitly rejected by the design
    document, so the threshold is always fitted on a held-out energy sample.
    """

    def __init__(
        self,
        *,
        mode: str = "quantile",
        quantile: float = 0.8,
        fdr_alpha: float = 0.1,
        momentum: float = 0.9,
        pollution_max: float = 0.3,
    ) -> None:
        super().__init__()
        if mode not in {"quantile", "fdr"}:
            raise ValueError("mode must be quantile or fdr")
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        if not 0.0 < fdr_alpha < 1.0:
            raise ValueError("fdr_alpha must be in (0, 1)")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        self.mode = mode
        self.quantile = float(quantile)
        self.fdr_alpha = float(fdr_alpha)
        self.momentum = float(momentum)
        self.pollution_max = float(pollution_max)
        self.register_buffer("threshold", torch.tensor(float("nan")))
        self.register_buffer("null_energy", torch.zeros(0))
        self.self_training_enabled = True

    @property
    def is_fitted(self) -> bool:
        return bool(torch.isfinite(self.threshold).item())

    def _quantile_threshold(self, energy: Tensor) -> Tensor:
        return torch.quantile(energy, self.quantile)

    def _fdr_threshold(self, energy: Tensor) -> Tensor:
        """BH threshold for low-energy discoveries under an empirical null."""

        if self.null_energy.numel() == 0:
            return self._quantile_threshold(energy)
        null = self.null_energy.to(energy)
        sorted_energy, _ = torch.sort(energy)
        count = sorted_energy.numel()
        # Trusted samples are low-energy, hence p_i = P_null(E <= e_i).
        null_counts = (
            null.reshape(1, -1) <= sorted_energy.reshape(-1, 1)
        ).sum(dim=1)
        p_values = (null_counts.to(energy.dtype) + 1.0) / (null.numel() + 1.0)
        ranks = torch.arange(1, count + 1, device=energy.device, dtype=energy.dtype)
        passing = p_values <= self.fdr_alpha * ranks / count
        if not bool(passing.any().item()):
            # A finite threshold below the minimum selects no pseudo labels.
            margin = sorted_energy[0].abs().clamp_min(1.0) * 1e-6
            return sorted_energy[0] - margin
        last = int(torch.nonzero(passing).max().item())
        return sorted_energy[last]

    @torch.no_grad()
    def fit(self, energy: Tensor, *, null_energy: Tensor | None = None) -> Tensor:
        """Fit the threshold on validation energies (called outside training steps)."""

        energy = energy.detach().reshape(-1)
        energy = energy[torch.isfinite(energy)]
        if energy.numel() == 0:
            return self.threshold
        if null_energy is not None:
            null = null_energy.detach().reshape(-1)
            self.null_energy = null[torch.isfinite(null)].to(self.null_energy.device)
        threshold = (
            self._quantile_threshold(energy)
            if self.mode == "quantile"
            else self._fdr_threshold(energy)
        ).to(self.threshold)
        if not self.is_fitted:
            self.threshold.copy_(threshold)
        else:
            momentum = self.momentum
            self.threshold.mul_(momentum).add_(threshold * (1.0 - momentum))
        return self.threshold

    @torch.no_grad()
    def select(self, energy: Tensor) -> Tensor:
        """Mask of trusted (low energy) samples usable as cold-start pseudo labels."""

        if not self.is_fitted:
            return torch.zeros_like(energy, dtype=torch.bool)
        return energy.detach() <= self.threshold

    @torch.no_grad()
    def report(self, energy: Tensor, is_true_positive: Tensor) -> PseudoLabelReport:
        """Precision/recall/pollution of energy-selected pseudo labels."""

        selected = self.select(energy)
        truth = is_true_positive.detach().bool()
        hit = float((selected & truth).sum().item())
        chosen = float(selected.sum().item())
        positives = float(truth.sum().item())
        precision = hit / chosen if chosen > 0 else 0.0
        recall = hit / positives if positives > 0 else 0.0
        pollution = 1.0 - precision if chosen > 0 else 1.0
        accepted = pollution <= self.pollution_max
        if not accepted:
            self.self_training_enabled = False
        return PseudoLabelReport(
            threshold=float(self.threshold.item()) if self.is_fitted else float("nan"),
            selected=int(chosen),
            precision=precision,
            recall=recall,
            pollution=pollution,
            accepted=accepted,
        )


# Shared-gradient conflict handling

VALID_MODES = {"none", "minnorm", "gradcraft", "multibalance"}


@dataclass
class ConflictStats:
    cos: float = 0.0
    norm_ratio: float = 0.0
    conflicted: bool = False
    auxiliary_scale: float = 1.0
    projected: bool = False
    descent_inner_product: float = 0.0
    per_group: dict[str, dict[str, float]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float]:
        return {
            "grad_conflict_cos": self.cos,
            "grad_norm_ratio": self.norm_ratio,
            "grad_conflict_ratio": float(self.conflicted),
            "grad_auxiliary_scale": self.auxiliary_scale,
            "grad_projected": float(self.projected),
            "grad_descent_inner_product": self.descent_inner_product,
        }


def _flatten(grads: list[Tensor | None], reference: list[Tensor]) -> Tensor:
    parts = [
        torch.zeros_like(ref).reshape(-1) if grad is None else grad.reshape(-1)
        for grad, ref in zip(grads, reference)
    ]
    if not parts:
        return torch.zeros(0)
    return torch.cat(parts)


def _unflatten(flat: Tensor, reference: list[Tensor]) -> list[Tensor]:
    outputs: list[Tensor] = []
    offset = 0
    for ref in reference:
        count = ref.numel()
        outputs.append(flat[offset : offset + count].reshape(ref.shape))
        offset += count
    return outputs


def parse_modes(mode: str) -> set[str]:
    """Parse ``"gradcraft+multibalance"`` style specifications."""

    tokens = {token.strip().lower() for token in str(mode).split("+") if token.strip()}
    if not tokens:
        tokens = {"none"}
    unknown = tokens - VALID_MODES
    if unknown:
        raise ValueError(f"unknown gradient modes: {sorted(unknown)}")
    if "none" in tokens and len(tokens) > 1:
        raise ValueError("mode 'none' cannot be combined with other modes")
    return tokens


class SharedGradientController:
    """Combine the recommendation gradient with the alignment gradient."""

    def __init__(self, mode: str = "none", *, norm_ratio_max: float = 1.0) -> None:
        self.modes = parse_modes(mode)
        self.mode = mode
        if norm_ratio_max <= 0:
            raise ValueError("norm_ratio_max must be positive")
        self.norm_ratio_max = float(norm_ratio_max)

    def combine_flat(self, primary: Tensor, auxiliary: Tensor) -> tuple[Tensor, ConflictStats]:
        stats = ConflictStats()
        if primary.numel() == 0:
            return primary, stats

        primary_norm = torch.linalg.vector_norm(primary).clamp_min(1e-12)
        auxiliary_norm = torch.linalg.vector_norm(auxiliary)
        cos = (primary @ auxiliary) / (primary_norm * auxiliary_norm.clamp_min(1e-12))
        stats.cos = float(cos)
        stats.norm_ratio = float(auxiliary_norm / primary_norm)
        stats.conflicted = stats.cos < 0.0

        adjusted = auxiliary
        if "multibalance" in self.modes and auxiliary_norm > 0:
            scale = torch.clamp(
                self.norm_ratio_max * primary_norm / auxiliary_norm, max=1.0
            )
            adjusted = adjusted * scale
            stats.auxiliary_scale = float(scale)

        if "gradcraft" in self.modes and stats.conflicted:
            before = torch.linalg.vector_norm(adjusted)
            adjusted = adjusted - (adjusted @ primary) / primary_norm.square() * primary
            after = torch.linalg.vector_norm(adjusted).clamp_min(1e-12)
            cap = torch.minimum(before, self.norm_ratio_max * primary_norm)
            adjusted = adjusted * (cap / after)
            stats.projected = True

        combined = primary + adjusted
        if "minnorm" in self.modes and stats.conflicted:
            difference = primary - adjusted
            denominator = (difference @ difference).clamp_min(1e-12)
            alpha = torch.clamp((adjusted - primary) @ adjusted / denominator, 0.0, 1.0)
            minnorm = alpha * primary + (1.0 - alpha) * adjusted
            minnorm_norm = torch.linalg.vector_norm(minnorm).clamp_min(1e-12)
            combined = minnorm * (primary_norm / minnorm_norm)
            stats.projected = True

        stats.descent_inner_product = float(combined @ primary)
        return combined, stats

    def combine(
        self,
        parameters: list[Tensor],
        primary_grads: list[Tensor | None],
        auxiliary_grads: list[Tensor | None],
        *,
        names: list[str] | None = None,
    ) -> tuple[list[Tensor], ConflictStats]:
        primary = _flatten(primary_grads, parameters)
        auxiliary = _flatten(auxiliary_grads, parameters)
        combined, stats = self.combine_flat(primary, auxiliary)
        if names is not None:
            stats.per_group = group_conflict_stats(
                parameters, primary_grads, auxiliary_grads, names
            )
        return _unflatten(combined, parameters), stats

    def apply(
        self,
        parameters: list[Tensor],
        primary_grads: list[Tensor | None],
        auxiliary_grads: list[Tensor | None],
        *,
        names: list[str] | None = None,
    ) -> ConflictStats:
        """Write the conflict-controlled gradient into ``parameter.grad``."""

        combined, stats = self.combine(
            parameters, primary_grads, auxiliary_grads, names=names
        )
        for parameter, gradient in zip(parameters, combined):
            if parameter.grad is None:
                parameter.grad = gradient.detach().clone()
            else:
                parameter.grad.copy_(gradient.detach())
        return stats


def group_conflict_stats(
    parameters: list[Tensor],
    primary_grads: list[Tensor | None],
    auxiliary_grads: list[Tensor | None],
    names: list[str],
) -> dict[str, dict[str, float]]:
    """Per-layer conflict matrix: ``cos(g_rec, g_align)`` and ``rho`` per group."""

    report: dict[str, dict[str, float]] = {}
    for name, parameter, primary, auxiliary in zip(
        names, parameters, primary_grads, auxiliary_grads
    ):
        primary_flat = (
            torch.zeros_like(parameter).reshape(-1) if primary is None else primary.reshape(-1)
        )
        auxiliary_flat = (
            torch.zeros_like(parameter).reshape(-1)
            if auxiliary is None
            else auxiliary.reshape(-1)
        )
        primary_norm = torch.linalg.vector_norm(primary_flat)
        auxiliary_norm = torch.linalg.vector_norm(auxiliary_flat)
        cos = float(
            (primary_flat @ auxiliary_flat)
            / (primary_norm.clamp_min(1e-12) * auxiliary_norm.clamp_min(1e-12))
        )
        report[name] = {
            "cos": cos,
            "norm_ratio": float(auxiliary_norm / primary_norm.clamp_min(1e-12)),
            "align_grad_norm": float(auxiliary_norm),
            "rec_grad_norm": float(primary_norm),
        }
    return report


# Diagnostic statistics

@dataclass
class NegativeTransferStats:
    full_tail_rate: float
    selective_tail_rate: float
    delta: float
    tail_count: int


def negative_transfer_rates(
    energy: Tensor, weights: Tensor, threshold: Tensor | float
) -> NegativeTransferStats:
    """Tail mass under uniform (full) versus energy-weighted (selective) transport."""

    energy = energy.detach().reshape(-1)
    weights = weights.detach().reshape(-1)
    finite = torch.isfinite(energy)
    energy = energy[finite]
    weights = weights[finite]
    if energy.numel() == 0:
        return NegativeTransferStats(0.0, 0.0, 0.0, 0)
    threshold_value = torch.as_tensor(threshold, dtype=energy.dtype, device=energy.device)
    if not bool(torch.isfinite(threshold_value).item()):
        return NegativeTransferStats(0.0, 0.0, 0.0, 0)
    tail = energy > threshold_value
    full_rate = float(tail.float().mean())
    weight_sum = weights.sum().clamp_min(1e-12)
    selective_rate = float(weights[tail].sum() / weight_sum)
    return NegativeTransferStats(
        full_tail_rate=full_rate,
        selective_tail_rate=selective_rate,
        delta=full_rate - selective_rate,
        tail_count=int(tail.sum()),
    )


class EnergyGainBuckets:
    """Accumulate (transfer score, target gain) pairs and bucket them by quantile."""

    def __init__(self, num_buckets: int = 4) -> None:
        if num_buckets < 2:
            raise ValueError("num_buckets must be at least 2")
        self.num_buckets = int(num_buckets)
        self._energy: list[Tensor] = []
        self._gain: list[Tensor] = []

    def reset(self) -> None:
        self._energy.clear()
        self._gain.clear()

    def update(self, energy: Tensor, gain: Tensor) -> None:
        energy = energy.detach().reshape(-1).cpu()
        gain = gain.detach().reshape(-1).cpu()
        if energy.numel() != gain.numel():
            raise ValueError("energy and gain must have the same number of samples")
        if energy.numel() == 0:
            return
        self._energy.append(energy)
        self._gain.append(gain)

    def report(self) -> dict[str, float]:
        if not self._energy:
            return {}
        energy = torch.cat(self._energy)
        gain = torch.cat(self._gain)
        order = torch.argsort(energy)
        chunks = torch.chunk(order, self.num_buckets)
        means = [float(gain[chunk].mean()) for chunk in chunks if chunk.numel() > 0]
        report = {f"energy_bucket_{index}_gain": value for index, value in enumerate(means)}
        report["energy_bucket_count"] = float(len(means))
        if len(means) >= 2:
            # Bucket 0 holds the lowest (most trustworthy) energy, so a healthy
            # mechanism yields a non-increasing gain sequence.
            report["energy_gain_monotone"] = float(
                all(means[i] >= means[i + 1] for i in range(len(means) - 1))
            )
            report["energy_gain_spread"] = means[0] - means[-1]
        return report


@dataclass
class How3Diagnostics:
    """Mutable container written by the model and read by training scripts."""

    stage: str = "none"
    energy_threshold: float = float("nan")
    negative_transfer: NegativeTransferStats = field(
        default_factory=lambda: NegativeTransferStats(0.0, 0.0, 0.0, 0)
    )
    trust_weight_entropy: float = 0.0
    align_mode: str = "w2"
    gradient: dict[str, float] = field(default_factory=dict)
    conflict_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    pseudo_label: dict[str, float] = field(default_factory=dict)
    buckets: dict[str, float] = field(default_factory=dict)
    isolation_ok: bool = True

    def as_dict(self) -> dict[str, float | str | bool]:
        payload: dict[str, float | str | bool] = {
            "stage": self.stage,
            "align_mode": self.align_mode,
            "energy_threshold": self.energy_threshold,
            "full_tail_neg_rate": self.negative_transfer.full_tail_rate,
            "selective_tail_neg_rate": self.negative_transfer.selective_tail_rate,
            "negative_transfer_delta": self.negative_transfer.delta,
            "tail_sample_count": float(self.negative_transfer.tail_count),
            "trust_weight_entropy": self.trust_weight_entropy,
            "private_isolation_ok": self.isolation_ok,
        }
        payload.update(self.gradient)
        payload.update(self.pseudo_label)
        payload.update(self.buckets)
        return payload


def trust_weight_entropy(weights: Tensor) -> float:
    """Normalized entropy of the trust weights; 1.0 means the gate is inactive."""

    weights = weights.detach().reshape(-1)
    if weights.numel() < 2:
        return 0.0
    probabilities = weights / weights.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy / torch.log(torch.tensor(float(weights.numel()))))


# Embedding visualization helpers

class VisualizationCore(Protocol):
    source_active_users: Tensor
    source_active_items: Tensor
    target_active_users: Tensor
    target_active_items: Tensor

    def __call__(self) -> tuple[Tensor, Tensor, Tensor, Tensor]: ...


def collect_active_embeddings(
    core: VisualizationCore,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return active source/target user and item embeddings for visualization."""

    source_users, source_items, target_users, target_items = core()
    return (
        source_users[core.source_active_users],
        source_items[core.source_active_items],
        target_users[core.target_active_users],
        target_items[core.target_active_items],
    )


def project_embedding_groups_2d(
    groups: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    """Project embedding groups through one shared PCA basis.

    A shared basis is essential: fitting each domain separately would make the
    resulting source/target positions visually incomparable.
    """

    if not groups:
        return ()
    matrices = []
    for group in groups:
        matrix = torch.as_tensor(group).detach().float().cpu()
        if matrix.ndim != 2:
            raise ValueError("each embedding group must be a rank-2 tensor")
        matrices.append(matrix)
    nonempty = [matrix for matrix in matrices if matrix.shape[0] > 0]
    if not nonempty:
        return tuple(matrix.new_empty((0, 2)) for matrix in matrices)
    feature_sizes = {matrix.shape[1] for matrix in nonempty}
    if len(feature_sizes) != 1:
        raise ValueError("all embedding groups must have the same feature size")

    combined = torch.cat(nonempty, dim=0)
    centered = combined - combined.mean(dim=0, keepdim=True)
    if centered.shape[0] < 2 or centered.shape[1] == 0:
        projected = centered.new_zeros((centered.shape[0], 2))
    else:
        _, _, vectors = torch.linalg.svd(centered, full_matrices=False)
        components = vectors[: min(2, vectors.shape[0])].transpose(0, 1)
        projected = centered @ components
        if projected.shape[1] < 2:
            projected = torch.nn.functional.pad(projected, (0, 2 - projected.shape[1]))

    outputs = []
    cursor = 0
    for matrix in matrices:
        count = matrix.shape[0]
        outputs.append(projected[cursor : cursor + count])
        cursor += count
    return tuple(outputs)


def save_active_embedding_plot(
    core: VisualizationCore,
    output_path: str | Path,
    *,
    max_points_per_group: int = 2000,
) -> Path:
    """Save a shared-PCA scatter plot for source/target users and items."""

    if max_points_per_group < 1:
        raise ValueError("max_points_per_group must be positive")
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "matplotlib is required to save embedding plots"
        ) from error

    groups = collect_active_embeddings(core)
    sampled = tuple(group[:max_points_per_group] for group in groups)
    projected = project_embedding_groups_2d(sampled)
    labels = ("source users", "source items", "target users", "target items")
    markers = ("o", "x", "o", "x")
    colors = ("tab:blue", "tab:cyan", "tab:orange", "tab:red")

    figure, axis = plt.subplots(figsize=(8, 6))
    for points, label, marker, color in zip(projected, labels, markers, colors):
        if points.shape[0] == 0:
            continue
        axis.scatter(
            points[:, 0].numpy(),
            points[:, 1].numpy(),
            s=10,
            alpha=0.65,
            marker=marker,
            color=color,
            label=label,
        )
    axis.set_title("HCTS-EMCDR active embeddings (shared PCA)")
    axis.set_xlabel("PC1")
    axis.set_ylabel("PC2")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


# HOW3 transfer controller

class How3Controller(nn.Module):
    """Validation-calibrated trust gate and shared-gradient controller."""

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        cross_weight: float = 1.0,
        threshold_mode: str = "quantile",
        threshold_quantile: float = 0.2,
        fdr_alpha: float = 0.1,
        hard_screen: bool = True,
        trust_max_fraction: float = 0.2,
        gradient_mode: str = "gradcraft+multibalance",
        gradient_norm_ratio_max: float = 0.5,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("How3 energy temperature must be positive")
        if cross_weight < 0:
            raise ValueError("How3 cross-energy weight must be non-negative")
        if not 0.0 < trust_max_fraction <= 1.0:
            raise ValueError("How3 trust_max_fraction must be in (0, 1]")
        self.temperature = float(temperature)
        self.cross_weight = float(cross_weight)
        self.hard_screen = bool(hard_screen)
        self.trust_max_fraction = float(trust_max_fraction)
        self.density_standardizer = RunningStandardizer(momentum=0.9)
        self.cross_standardizer = RunningStandardizer(momentum=0.9)
        self.calibrator = EnergyThresholdCalibrator(
            mode=threshold_mode,
            quantile=threshold_quantile,
            fdr_alpha=fdr_alpha,
            momentum=0.0,
        )
        self.gradient_controller = SharedGradientController(
            gradient_mode, norm_ratio_max=gradient_norm_ratio_max
        )
        self.diagnostics = How3Diagnostics(align_mode="energy_weighted_distribution_sw2")

    @property
    def is_calibrated(self) -> bool:
        return self.calibrator.is_fitted

    def score(self, density: Tensor, cross: Tensor) -> Tensor:
        density = self.density_standardizer.transform(density)
        cross = self.cross_standardizer.transform(cross)
        return density + self.cross_weight * cross

    @torch.no_grad()
    def fit(
        self,
        density: Tensor,
        cross: Tensor,
        *,
        null_density: Tensor | None = None,
        null_cross: Tensor | None = None,
    ) -> float:
        finite = torch.isfinite(density) & torch.isfinite(cross)
        density = density.detach().reshape(-1)[finite.reshape(-1)]
        cross = cross.detach().reshape(-1)[finite.reshape(-1)]
        if density.numel() == 0:
            return float(self.calibrator.threshold)
        self.density_standardizer.fit(density)
        self.cross_standardizer.fit(cross)
        energy = self.score(density, cross)
        null_energy = None
        if null_density is not None and null_cross is not None:
            null_finite = torch.isfinite(null_density) & torch.isfinite(null_cross)
            null_density = null_density.detach().reshape(-1)[null_finite.reshape(-1)]
            null_cross = null_cross.detach().reshape(-1)[null_finite.reshape(-1)]
            if null_density.numel() > 0:
                null_energy = self.score(null_density, null_cross)
        threshold = self.calibrator.fit(energy, null_energy=null_energy)
        self.diagnostics.energy_threshold = float(threshold)
        return float(threshold)

    def trusted_weights(self, energy: Tensor) -> tuple[Tensor, Tensor]:
        weights = trust_weights(energy, self.temperature)
        if not self.is_calibrated:
            return weights, torch.ones_like(energy, dtype=torch.bool)
        trusted = self.calibrator.select(energy)
        cap = max(1, int(energy.numel() * self.trust_max_fraction))
        if int(trusted.sum()) > cap:
            selected = torch.topk(energy, cap, largest=False).indices
            trusted = torch.zeros_like(trusted)
            trusted[selected] = True
        if self.hard_screen:
            weights = weights * trusted.to(weights.dtype)
            if float(weights.sum()) <= 0:
                selected = torch.argmin(energy)
                weights = torch.zeros_like(weights)
                weights[selected] = 1.0
                trusted[selected] = True
            weights = weights / weights.sum().clamp_min(1e-12)
        stats = negative_transfer_rates(energy, weights, self.calibrator.threshold)
        self.diagnostics.negative_transfer = stats
        self.diagnostics.trust_weight_entropy = trust_weight_entropy(weights)
        self.diagnostics.pseudo_label = {
            "trusted_transfer_count": float(trusted.sum()),
            "trusted_transfer_fraction": float(trusted.float().mean()),
        }
        return weights, trusted

    @staticmethod
    def alignment(
        source: Tensor,
        destination: Tensor,
        source_weights: Tensor,
        destination_weights: Tensor,
        *,
        num_projections: int,
    ) -> Tensor:
        return weighted_sliced_wasserstein2(
            source,
            destination,
            source_weights,
            destination_weights,
            num_projections=num_projections,
        )


# Model outputs

@dataclass
class LossBreakdown:
    total: Tensor
    source: Tensor
    target: Tensor
    transfer: Tensor
    hierarchy: Tensor
    gate_balance: Tensor
    disentanglement: Tensor
    density_energy: Tensor
    clustering: Tensor
    selective_align: Tensor

    @property
    def energy(self) -> Tensor:
        """Compatibility alias for the How2 density-energy term."""

        return self.density_energy


@dataclass
class How123Losses:
    primary: Tensor
    align: Tensor
    breakdown: LossBreakdown


@dataclass
class ModelRepresentations:
    source_users: Tensor
    source_items: Tensor
    target_users: Tensor
    target_items: Tensor
    source_user_gates: Tensor | None
    source_item_gates: Tensor | None
    target_user_gates: Tensor | None
    target_item_gates: Tensor | None
    source_streams: DualStreamOutput | None
    target_streams: DualStreamOutput | None
    source_base_users: Tensor
    target_base_users: Tensor

    def encoded(self):
        return (
            self.source_users,
            self.source_items,
            self.target_users,
            self.target_items,
            self.source_user_gates,
            self.source_item_gates,
            self.target_user_gates,
            self.target_item_gates,
        )




















# Recommendation model

class HCTSEmCDR(nn.Module):
    """Corrected and dependency-light realization of HCTS."""

    def __init__(
        self,
        *,
        total_num_users: int,
        total_num_items: int,
        target_num_users: int,
        target_num_items: int,
        overlapped_num_users: int,
        overlapped_num_items: int,
        source_edges: tuple[Tensor, Tensor],
        target_edges: tuple[Tensor, Tensor],
        embedding_size: int = 64,
        num_layers: int = 2,
        source_curvature: float = 1.0,
        target_curvature: float = 1.0,
        learnable_curvature: bool = True,
        margin: float = 0.1,
        temperature: float = 0.2,
        transfer_weight: float = 0.05,
        hierarchy_weight: float = 0.01,
        hierarchy_radius_sq: float = 10.0,
        reg_weight: float = 1e-4,
        num_transfer_negatives: int = 3,
        graph_dropout: float = 0.0,
        mapping_dropout: float = 0.0,
        bidirectional: bool = True,
        enable_user_user: bool = True,
        enable_user_item: bool = True,
        enable_item_user: bool = True,
        enable_item_item: bool = True,
        detach_transferred: bool = True,
        init_scale: float = 0.05,
        geometry_mode: str = "lorentz",
        manifold_dimensions: tuple[int, int, int] | list[int] | None = None,
        source_spherical_curvature: float = 1.0,
        target_spherical_curvature: float = 1.0,
        source_hyperbolic_curvature: float | None = None,
        target_hyperbolic_curvature: float | None = None,
        gate_hidden_size: int = 32,
        gate_heads: int = 4,
        gate_temperature: float = 0.5,
        topology_gate_strength: float = 1.0,
        energy_gate_strength: float = 1.0,
        gate_dropout: float = 0.0,
        energy_chunk_size: int = 262144,
        contrastive_chunk_size: int = 512,
        full_sort_item_chunk_size: int = 4096,
        loss_type: str = "margin",
        normalize_energy_by_degree: bool = True,
        normalize_energy_per_node: bool = True,
        detach_energy_feedback: bool = True,
        max_transfer_users: int | None = None,
        normalize_component_distances: bool = False,
        gate_warmup_epochs: int = 0,
        gate_prior_epochs: int = 0,
        gate_ramp_epochs: int | None = None,
        gate_initial_adaptive_scale: float = 0.0,
        equal_component_prior: bool = False,
        gate_shadow_train_during_prior: bool = False,
        gate_floor: float = 0.0,
        gate_balance_weight: float = 0.0,
        gate_node_kl_ratio: float = 0.0,
        enable_how2: bool = False,
        how2_hidden_size: int = 64,
        disentanglement_weight: float = 0.001,
        density_energy_weight: float = 0.001,
        cluster_weight: float = 0.001,
        density_energy_temperature: float = 1.0,
        density_energy_margin: float = 1.0,
        density_energy_noise_scale: float = 2.0,
        density_energy_l2: float = 0.001,
        num_preference_clusters: int = 4,
        cluster_temperature: float = 0.2,
        cluster_balance_weight: float = 0.1,
        cluster_standardize_distances: bool = False,
        cluster_distance_scale_floor: float = 1.0,
        multi_interest_temperature: float = 0.2,
        interest_separation_weight: float = 0.1,
        interest_separation_margin: float = 1.0,
        w2_num_projections: int = 16,
        how2_warmup_epochs: int = 0,
        how2_ramp_epochs: int = 1,
        how2_residual_scale: float = 1.0,
        how2_fixed_ramp_scale: float | None = None,
        how2_initial_ramp_scale: float = 0.0,
        detach_how2_auxiliary_base: bool = True,
        freeze_how1_after_warmup: bool = False,
        enable_how3: bool = False,
        selective_align_weight: float = 0.0,
        selective_contrastive_weight: float = 0.0,
        align_loss_cap_ratio: float = 0.0,
        how3_start_epoch: int = 0,
        how3_ramp_epochs: int = 0,
        how3_initial_scale: float = 0.0,
        energy_temperature: float = 1.0,
        energy_cross_weight: float = 1.0,
        energy_threshold_mode: str = "quantile",
        energy_threshold_quantile: float = 0.2,
        energy_fdr_alpha: float = 0.1,
        energy_hard_screen: bool = True,
        energy_trust_max_fraction: float = 0.2,
        align_detach_source: bool = True,
        shared_grad_mode: str = "gradcraft+multibalance",
        shared_grad_norm_ratio_max: float = 0.5,
        graph_encoder: str = "hcts",
    ) -> None:
        super().__init__()
        if total_num_users <= 1 or total_num_items <= 1:
            raise ValueError("entity counts must include PAD and at least one real entity")
        if overlapped_num_users <= 1:
            raise ValueError("HCTS requires at least one overlapping user in addition to PAD")
        if temperature <= 0 or num_transfer_negatives <= 0:
            raise ValueError("temperature and num_transfer_negatives must be positive")
        if min(
            disentanglement_weight,
            density_energy_weight,
            cluster_weight,
            density_energy_l2,
        ) < 0:
            raise ValueError("How2 loss weights must be non-negative")
        if density_energy_temperature <= 0 or density_energy_noise_scale <= 0:
            raise ValueError(
                "density-energy temperature and noise scale must be positive"
            )
        if multi_interest_temperature <= 0:
            raise ValueError("multi-interest temperature must be positive")
        if cluster_distance_scale_floor <= 0:
            raise ValueError("cluster distance scale floor must be positive")
        if interest_separation_weight < 0 or interest_separation_margin < 0:
            raise ValueError("interest separation settings must be non-negative")
        if how2_warmup_epochs < 0 or how2_ramp_epochs < 1:
            raise ValueError(
                "How2 warmup must be non-negative and ramp must be positive"
            )
        if not 0.0 <= how2_residual_scale <= 1.0:
            raise ValueError("how2_residual_scale must be in [0,1]")
        if how2_fixed_ramp_scale is not None and not 0.0 <= how2_fixed_ramp_scale <= 1.0:
            raise ValueError("how2_fixed_ramp_scale must be in [0,1]")
        if not 0.0 <= how2_initial_ramp_scale <= 1.0:
            raise ValueError("how2_initial_ramp_scale must be in [0,1]")
        if align_loss_cap_ratio < 0:
            raise ValueError("align_loss_cap_ratio must be non-negative")
        if enable_how3 and not enable_how2:
            raise ValueError("How3 requires the How2 shared-density stream")
        if selective_align_weight < 0 or selective_contrastive_weight < 0:
            raise ValueError("How3 alignment weights must be non-negative")
        if how3_start_epoch < 0:
            raise ValueError("how3_start_epoch must be non-negative")
        if how3_ramp_epochs < 0:
            raise ValueError("how3_ramp_epochs must be non-negative")
        if not 0.0 <= how3_initial_scale <= 1.0:
            raise ValueError("how3_initial_scale must be in [0,1]")

        self.total_num_users = int(total_num_users)
        self.total_num_items = int(total_num_items)
        self.target_num_users = int(target_num_users)
        self.target_num_items = int(target_num_items)
        self.overlapped_num_users = int(overlapped_num_users)
        self.overlapped_num_items = int(overlapped_num_items)
        self.embedding_size = int(embedding_size)
        self.graph_encoder_mode = str(graph_encoder).lower()
        if self.graph_encoder_mode != "hcts":
            raise ValueError("graph_encoder must be hcts")
        self.margin = float(margin)
        self.temperature = float(temperature)
        self.transfer_weight = float(transfer_weight)
        self.hierarchy_weight = float(hierarchy_weight)
        self.hierarchy_radius_sq = float(hierarchy_radius_sq)
        self.reg_weight = float(reg_weight)
        self.num_transfer_negatives = int(num_transfer_negatives)
        self.bidirectional = bool(bidirectional)
        self.detach_transferred = bool(detach_transferred)
        self.enabled_relations = {
            "uu": bool(enable_user_user),
            "ui": bool(enable_user_item),
            "iu": bool(enable_item_user),
            "ii": bool(enable_item_item),
        }
        self.init_scale = float(init_scale)
        self.geometry_mode = str(geometry_mode).lower()
        if self.geometry_mode not in {"lorentz", "product"}:
            raise ValueError("geometry_mode must be lorentz or product")
        self.uses_product_manifold = self.geometry_mode == "product"
        self._evaluation_cache = None
        self.energy_chunk_size = int(energy_chunk_size)
        self.contrastive_chunk_size = int(contrastive_chunk_size)
        self.full_sort_item_chunk_size = int(full_sort_item_chunk_size)
        if min(
            self.energy_chunk_size,
            self.contrastive_chunk_size,
            self.full_sort_item_chunk_size,
        ) < 1:
            raise ValueError("memory chunk sizes must be positive")
        self.loss_type = str(loss_type).lower()
        if self.loss_type not in {"bpr", "margin", "hinge"}:
            raise ValueError("loss_type must be BPR, margin, or hinge")
        self.normalize_energy_by_degree = bool(normalize_energy_by_degree)
        self.normalize_energy_per_node = bool(normalize_energy_per_node)
        self.detach_energy_feedback = bool(detach_energy_feedback)
        self.normalize_component_distances = bool(normalize_component_distances)
        self.gate_warmup_epochs = int(gate_warmup_epochs)
        self.gate_prior_epochs = int(gate_prior_epochs)
        self.gate_ramp_epochs = int(
            gate_warmup_epochs if gate_ramp_epochs is None else gate_ramp_epochs
        )
        self.gate_initial_adaptive_scale = float(gate_initial_adaptive_scale)
        self.equal_component_prior = bool(equal_component_prior)
        self.gate_shadow_train_during_prior = bool(
            gate_shadow_train_during_prior
        )
        self.gate_floor = float(gate_floor)
        self.gate_balance_weight = float(gate_balance_weight)
        self.gate_node_kl_ratio = float(gate_node_kl_ratio)
        self.enable_how2 = bool(enable_how2)
        self.disentanglement_weight = float(disentanglement_weight)
        self.density_energy_weight = float(density_energy_weight)
        self.cluster_weight = float(cluster_weight)
        self.density_energy_temperature = float(density_energy_temperature)
        self.density_energy_margin = float(density_energy_margin)
        self.density_energy_noise_scale = float(density_energy_noise_scale)
        self.density_energy_l2 = float(density_energy_l2)
        self.multi_interest_temperature = float(multi_interest_temperature)
        self.interest_separation_weight = float(interest_separation_weight)
        self.interest_separation_margin = float(interest_separation_margin)
        self.distribution_num_projections = int(w2_num_projections)
        self.how2_warmup_epochs = int(how2_warmup_epochs)
        self.how2_ramp_epochs = int(how2_ramp_epochs)
        self.how2_residual_scale = float(how2_residual_scale)
        self.how2_fixed_ramp_scale = (
            None if how2_fixed_ramp_scale is None else float(how2_fixed_ramp_scale)
        )
        self.how2_initial_ramp_scale = float(how2_initial_ramp_scale)
        self.detach_how2_auxiliary_base = bool(detach_how2_auxiliary_base)
        self.freeze_how1_after_warmup = bool(freeze_how1_after_warmup)
        self._how2_ramp_scale = 0.0 if self.enable_how2 else 1.0
        self._training_epoch = 0
        self._how1_frozen = False
        self.enable_how3 = bool(enable_how3)
        self.selective_align_weight = float(selective_align_weight)
        self.selective_contrastive_weight = float(selective_contrastive_weight)
        self.align_loss_cap_ratio = float(align_loss_cap_ratio)
        self.how3_start_epoch = int(how3_start_epoch)
        self.how3_ramp_epochs = int(how3_ramp_epochs)
        self.how3_initial_scale = float(how3_initial_scale)
        self.align_detach_source = bool(align_detach_source)
        if self.gate_warmup_epochs < 0:
            raise ValueError("gate_warmup_epochs must be non-negative")
        if self.gate_prior_epochs < 0 or self.gate_ramp_epochs < 0:
            raise ValueError("gate prior/ramp epochs must be non-negative")
        if not 0.0 <= self.gate_initial_adaptive_scale <= 1.0:
            raise ValueError("gate_initial_adaptive_scale must be in [0,1]")
        if not 0.0 <= self.gate_floor < 1.0:
            raise ValueError("gate_floor must be in [0, 1)")
        if self.gate_balance_weight < 0.0:
            raise ValueError("gate_balance_weight must be non-negative")
        if self.gate_node_kl_ratio < 0.0:
            raise ValueError("gate_node_kl_ratio must be non-negative")
        self.max_transfer_users = (
            None if max_transfer_users is None else int(max_transfer_users)
        )
        if self.max_transfer_users is not None and self.max_transfer_users < 1:
            raise ValueError("max_transfer_users must be positive or None")

        source_hyperbolic_curvature = (
            source_curvature
            if source_hyperbolic_curvature is None
            else source_hyperbolic_curvature
        )
        target_hyperbolic_curvature = (
            target_curvature
            if target_hyperbolic_curvature is None
            else target_hyperbolic_curvature
        )
        self.source_curvature_module = PositiveCurvature(
            source_hyperbolic_curvature, learnable=learnable_curvature
        )
        self.target_curvature_module = PositiveCurvature(
            target_hyperbolic_curvature, learnable=learnable_curvature
        )
        if self.uses_product_manifold:
            dimensions = split_component_dimensions(embedding_size, manifold_dimensions)
            self.product_geometry = ProductManifoldGeometry(
                dimensions,
                normalize_component_distances=self.normalize_component_distances,
            )
            component_prior = (
                [1.0 / 3.0] * 3
                if self.equal_component_prior
                else [value / sum(dimensions) for value in dimensions]
            )
            self.source_spherical_curvature_module = PositiveCurvature(
                source_spherical_curvature, learnable=learnable_curvature
            )
            self.target_spherical_curvature_module = PositiveCurvature(
                target_spherical_curvature, learnable=learnable_curvature
            )
            self.source_gate = TopologyEnergyGate(
                hidden_size=gate_hidden_size,
                heads=gate_heads,
                temperature=gate_temperature,
                topology_strength=topology_gate_strength,
                energy_strength=energy_gate_strength,
                dropout=gate_dropout,
                component_prior=component_prior,
                gate_floor=self.gate_floor,
                shadow_train_during_prior=self.gate_shadow_train_during_prior,
            )
            self.target_gate = TopologyEnergyGate(
                hidden_size=gate_hidden_size,
                heads=gate_heads,
                temperature=gate_temperature,
                topology_strength=topology_gate_strength,
                energy_strength=energy_gate_strength,
                dropout=gate_dropout,
                component_prior=component_prior,
                gate_floor=self.gate_floor,
                shadow_train_during_prior=self.gate_shadow_train_during_prior,
            )
            self.set_training_epoch(0)

        self.source_user_embedding = nn.Embedding(
            self.total_num_users, self.embedding_size, padding_idx=0
        )
        self.source_item_embedding = nn.Embedding(
            self.total_num_items, self.embedding_size, padding_idx=0
        )
        self.target_user_embedding = nn.Embedding(
            self.total_num_users, self.embedding_size, padding_idx=0
        )
        self.target_item_embedding = nn.Embedding(
            self.total_num_items, self.embedding_size, padding_idx=0
        )
        self.source_encoder = HCTSGraphEncoder(num_layers, graph_dropout)
        self.target_encoder = HCTSGraphEncoder(num_layers, graph_dropout)
        if self.uses_product_manifold:
            self.source_to_target = ProductManifoldLinear(
                self.product_geometry, dropout=mapping_dropout
            )
            self.target_to_source = ProductManifoldLinear(
                self.product_geometry, dropout=mapping_dropout
            )
        else:
            self.source_to_target = LorentzLinear(
                self.embedding_size, self.embedding_size, dropout=mapping_dropout
            )
            self.target_to_source = LorentzLinear(
                self.embedding_size, self.embedding_size, dropout=mapping_dropout
            )

        how2_rng_state = (
            torch.random.get_rng_state() if self.enable_how2 else None
        )
        if self.enable_how2:
            component_dimensions = (
                self.product_geometry.dimensions
                if self.uses_product_manifold
                else (self.embedding_size,)
            )
            self.disentangler = SharedPrivateDisentangler(
                self.embedding_size, int(how2_hidden_size),
                component_dimensions=component_dimensions,
            )
            self.shared_density_energy = ManifoldDensityEnergy(
                self.embedding_size, int(how2_hidden_size), component_dimensions
            )
            self.source_private_density_energy = ManifoldDensityEnergy(
                self.embedding_size, int(how2_hidden_size), component_dimensions
            )
            self.target_private_density_energy = ManifoldDensityEnergy(
                self.embedding_size, int(how2_hidden_size), component_dimensions
            )
            self.wasserstein_independence = SlicedWassersteinIndependence(
                self.embedding_size, int(w2_num_projections)
            )
            cluster_arguments = {
                "dimension": self.embedding_size,
                "num_clusters": int(num_preference_clusters),
                "temperature": float(cluster_temperature),
                "balance_weight": float(cluster_balance_weight),
                "standardize_distances": bool(cluster_standardize_distances),
                "distance_scale_floor": float(cluster_distance_scale_floor),
            }
            self.shared_clusterer = EntropyManifoldClusterer(**cluster_arguments)
            self.source_private_clusterer = EntropyManifoldClusterer(
                **cluster_arguments
            )
            self.target_private_clusterer = EntropyManifoldClusterer(
                **cluster_arguments
            )
        else:
            self.disentangler = None
            self.shared_density_energy = None
            self.source_private_density_energy = None
            self.target_private_density_energy = None
            self.wasserstein_independence = None
            self.shared_clusterer = None
            self.source_private_clusterer = None
            self.target_private_clusterer = None
        self.how3 = (
            How3Controller(
                temperature=energy_temperature,
                cross_weight=energy_cross_weight,
                threshold_mode=energy_threshold_mode,
                threshold_quantile=energy_threshold_quantile,
                fdr_alpha=energy_fdr_alpha,
                hard_screen=energy_hard_screen,
                trust_max_fraction=energy_trust_max_fraction,
                gradient_mode=shared_grad_mode,
                gradient_norm_ratio_max=shared_grad_norm_ratio_max,
            )
            if self.enable_how3 else None
        )
        if how2_rng_state is not None:
            # Module constructors initialize parameters eagerly. Restore the
            # RNG so enabling How2 cannot change any common How1 parameter.
            torch.random.set_rng_state(how2_rng_state)

        source_users, source_items = source_edges
        target_users, target_items = target_edges
        source_offsets, source_neighbors = build_neighbor_csr(
            source_users, source_items, num_users=self.total_num_users
        )
        target_offsets, target_neighbors = build_neighbor_csr(
            target_users, target_items, num_users=self.total_num_users
        )
        self.register_buffer("source_neighbor_offsets", source_offsets)
        self.register_buffer("source_neighbor_items", source_neighbors)
        self.register_buffer("target_neighbor_offsets", target_offsets)
        self.register_buffer("target_neighbor_items", target_neighbors)
        self.register_buffer(
            "source_adjacency",
            build_normalized_bipartite_adjacency(
                source_users,
                source_items,
                num_users=self.total_num_users,
                num_items=self.total_num_items,
            ),
        )
        self.register_buffer(
            "target_adjacency",
            build_normalized_bipartite_adjacency(
                target_users,
                target_items,
                num_users=self.total_num_users,
                num_items=self.total_num_items,
            ),
        )
        if self.uses_product_manifold:
            self.register_buffer(
                "source_topology_features", build_topology_features(self.source_adjacency)
            )
            self.register_buffer(
                "target_topology_features", build_topology_features(self.target_adjacency)
            )
            self.register_buffer(
                "source_node_degree",
                torch.bincount(
                    self.source_adjacency.indices()[0],
                    minlength=self.source_adjacency.size(0),
                ).to(self.source_adjacency.dtype),
            )
            self.register_buffer(
                "target_node_degree",
                torch.bincount(
                    self.target_adjacency.indices()[0],
                    minlength=self.target_adjacency.size(0),
                ).to(self.target_adjacency.dtype),
            )
        self.register_buffer("source_item_catalog", torch.unique(source_items.long()))
        self.register_buffer("target_item_catalog", torch.unique(target_items.long()))
        self.register_buffer("source_active_users", torch.unique(source_users.long()))
        self.register_buffer("source_active_items", torch.unique(source_items.long()))
        self.register_buffer("target_active_users", torch.unique(target_users.long()))
        self.register_buffer("target_active_items", torch.unique(target_items.long()))
        self.reset_parameters()

    @property
    def source_curvature(self) -> Tensor:
        return self.source_curvature_module()

    @property
    def target_curvature(self) -> Tensor:
        return self.target_curvature_module()

    @property
    def source_spherical_curvature(self) -> Tensor:
        if not self.uses_product_manifold:
            raise AttributeError("spherical curvature is only defined in product mode")
        return self.source_spherical_curvature_module()

    @property
    def target_spherical_curvature(self) -> Tensor:
        if not self.uses_product_manifold:
            raise AttributeError("spherical curvature is only defined in product mode")
        return self.target_spherical_curvature_module()

    def reset_parameters(self) -> None:
        for embedding in (
            self.source_user_embedding,
            self.source_item_embedding,
            self.target_user_embedding,
            self.target_item_embedding,
        ):
            nn.init.normal_(embedding.weight, std=self.init_scale)
        self.source_to_target.reset_parameters()
        self.target_to_source.reset_parameters()
        if self.enable_how2:
            self.disentangler.shared_branch.reset_parameters()
            for branch in self.disentangler.private_branches.values():
                branch.reset_parameters()
            self.shared_density_energy.reset_parameters()
            self.source_private_density_energy.reset_parameters()
            self.target_private_density_energy.reset_parameters()
            self.shared_clusterer.reset_parameters()
            self.source_private_clusterer.reset_parameters()
            self.target_private_clusterer.reset_parameters()
        with torch.no_grad():
            self.source_user_embedding.weight[
                self.overlapped_num_users : self.target_num_users
            ].zero_()
            self.source_item_embedding.weight[
                self.overlapped_num_items : self.target_num_items
            ].zero_()
            self.target_user_embedding.weight[self.target_num_users :].zero_()
            self.target_item_embedding.weight[self.target_num_items :].zero_()
            for embedding in (
                self.source_user_embedding,
                self.source_item_embedding,
                self.target_user_embedding,
                self.target_item_embedding,
            ):
                embedding.weight[0].zero_()

    def clear_evaluation_cache(self) -> None:
        self._evaluation_cache = None

    def train(self, mode: bool = True):
        if mode:
            self.clear_evaluation_cache()
        return super().train(mode)

    def set_training_epoch(self, epoch: int) -> None:
        """Update How1 gate and staged How2 schedules from the trainer callback."""

        self.clear_evaluation_cache()
        epoch = max(int(epoch), 0)
        self._training_epoch = epoch
        if self.uses_product_manifold:
            if self.gate_prior_epochs == 0 and self.gate_ramp_epochs == 0:
                gate_scale = 1.0
            elif epoch <= self.gate_prior_epochs:
                gate_scale = self.gate_initial_adaptive_scale
            elif self.gate_ramp_epochs == 0:
                gate_scale = 1.0
            else:
                gate_progress = min(
                    (epoch - self.gate_prior_epochs) / self.gate_ramp_epochs,
                    1.0,
                )
                gate_scale = self.gate_initial_adaptive_scale + (
                    1.0 - self.gate_initial_adaptive_scale
                ) * gate_progress
            self.source_gate.set_adaptive_scale(gate_scale)
            self.target_gate.set_adaptive_scale(gate_scale)
        if self.enable_how2:
            self._how2_ramp_scale = (
                self.how2_fixed_ramp_scale
                if self.how2_fixed_ramp_scale is not None
                else self.how2_initial_ramp_scale
                + (1.0 - self.how2_initial_ramp_scale)
                * min(
                    max(epoch - self.how2_warmup_epochs, 0)
                    / self.how2_ramp_epochs,
                    1.0,
                )
            )
            should_freeze = (
                self.freeze_how1_after_warmup
                and epoch > self.how2_warmup_epochs
            )
            self._set_how1_frozen(should_freeze)
        if self.how3_active and self._how1_frozen:
            for parameter in self.source_to_target.parameters():
                parameter.requires_grad_(True)

    @property
    def how3_active(self) -> bool:
        return bool(
            self.enable_how3
            and self.selective_align_weight > 0
            and self._training_epoch >= self.how3_start_epoch
        )

    @property
    def how3_loss_scale(self) -> float:
        if not self.how3_active:
            return 0.0
        if self.how3_ramp_epochs == 0:
            return 1.0
        progress = min(
            max(self._training_epoch - self.how3_start_epoch, 0)
            / self.how3_ramp_epochs,
            1.0,
        )
        return self.how3_initial_scale + (1.0 - self.how3_initial_scale) * progress

    @property
    def effective_selective_align_weight(self) -> float:
        return self.selective_align_weight * self.how3_loss_scale

    @staticmethod
    def _is_how2_parameter(name: str) -> bool:
        """Return whether a parameter belongs exclusively to the How2 branch."""

        return name.startswith(
            (
                "disentangler.",
                "shared_density_energy.",
                "source_private_density_energy.",
                "target_private_density_energy.",
                "wasserstein_independence.",
                "shared_clusterer.",
                "source_private_clusterer.",
                "target_private_clusterer.",
            )
        )

    def _set_how1_frozen(self, frozen: bool) -> None:
        """Freeze the mature How1 base while leaving all How2 heads trainable."""

        frozen = bool(frozen)
        if self._how1_frozen == frozen:
            return
        for name, parameter in self.named_parameters():
            if not self._is_how2_parameter(name):
                parameter.requires_grad_(not frozen)
        self._how1_frozen = frozen

    def training_schedule_state(self) -> dict[str, int | float | bool]:
        """Serializable non-parameter state needed to reproduce validation exactly."""

        return {
            "training_epoch": self._training_epoch,
            "how2_ramp_scale": self._how2_ramp_scale,
            "how1_frozen": self._how1_frozen,
            "gate_adaptive_scale": self.gate_adaptive_scale,
            "gate_prior_epochs": self.gate_prior_epochs,
            "gate_ramp_epochs": self.gate_ramp_epochs,
            "how3_loss_scale": self.how3_loss_scale,
            "how3_active": self.how3_active,
        }

    def load_training_schedule_state(self, state: dict | None) -> None:
        """Restore the schedule paired with a checkpoint before evaluation."""

        if not state:
            return
        self.clear_evaluation_cache()
        self._training_epoch = int(state.get("training_epoch", 0))
        if self.enable_how2:
            self._how2_ramp_scale = float(state.get("how2_ramp_scale", 0.0))
            self._set_how1_frozen(bool(state.get("how1_frozen", False)))
        if self.uses_product_manifold:
            gate_scale = float(state.get("gate_adaptive_scale", 1.0))
            self.source_gate.set_adaptive_scale(gate_scale)
            self.target_gate.set_adaptive_scale(gate_scale)

    @property
    def how2_adaptive_scale(self) -> float:
        if not self.enable_how2:
            return 0.0
        return self.how2_residual_scale * self._how2_ramp_scale

    @property
    def how2_loss_scale(self) -> float:
        if not self.enable_how2:
            return 0.0
        return self._how2_ramp_scale

    @property
    def gate_adaptive_scale(self) -> float:
        if not self.uses_product_manifold:
            return 1.0
        return self.source_gate.adaptive_scale

    def _gate_balance_loss(
        self,
        source_user_gates: Tensor | None,
        source_item_gates: Tensor | None,
        target_user_gates: Tensor | None,
        target_item_gates: Tensor | None,
    ) -> Tensor:
        """Prevent population-level component collapse while preserving node diversity."""

        if not self.uses_product_manifold:
            return self.source_user_embedding.weight.sum() * 0.0
        if (
            self.gate_adaptive_scale == 0.0
            and self.gate_shadow_train_during_prior
            and self.source_gate.latest_learned_gates is not None
            and self.target_gate.latest_learned_gates is not None
        ):
            source_user_gates, source_item_gates = torch.split(
                self.source_gate.latest_learned_gates,
                [self.total_num_users, self.total_num_items],
            )
            target_user_gates, target_item_gates = torch.split(
                self.target_gate.latest_learned_gates,
                [self.total_num_users, self.total_num_items],
            )
        groups = (
            source_user_gates[self.source_active_users],
            source_item_gates[self.source_active_items],
            target_user_gates[self.target_active_users],
            target_item_gates[self.target_active_items],
        )
        prior = self.source_gate.component_prior
        population_losses = []
        node_losses = []
        for gates in groups:
            safe_gates = gates.clamp_min(1e-12)
            mean_gate = safe_gates.mean(dim=0)
            population_losses.append(
                (mean_gate * (mean_gate.log() - prior.log())).sum()
            )
            if self.gate_node_kl_ratio > 0.0:
                node_losses.append(
                    (
                        safe_gates
                        * (safe_gates.log() - prior.log().unsqueeze(0))
                    ).sum(dim=-1).mean()
                )
        loss = torch.stack(population_losses).mean()
        if node_losses:
            loss = loss + self.gate_node_kl_ratio * torch.stack(node_losses).mean()
        return loss

    def _domain_curvatures(self, domain: str) -> tuple[Tensor, Tensor]:
        if domain == "source":
            return self.source_spherical_curvature, self.source_curvature
        if domain == "target":
            return self.target_spherical_curvature, self.target_curvature
        raise ValueError("domain must be source or target")

    def _encode_domain(
        self,
        user_embedding: nn.Embedding,
        item_embedding: nn.Embedding,
        encoder: HCTSGraphEncoder,
        adjacency: Tensor,
        domain: str,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        tangent = torch.cat([user_embedding.weight, item_embedding.weight], dim=0)
        tangent = encoder(tangent, adjacency)
        if self.uses_product_manifold:
            spherical_curvature, hyperbolic_curvature = self._domain_curvatures(domain)
            points = self.product_geometry.expmap0(
                tangent, spherical_curvature, hyperbolic_curvature
            )
            energy_context = (
                torch.no_grad() if self.detach_energy_feedback else nullcontext()
            )
            with energy_context:
                raw_energy = self.product_geometry.structural_energy(
                    points,
                    adjacency.indices(),
                    spherical_curvature,
                    hyperbolic_curvature,
                    chunk_size=self.energy_chunk_size,
                )
                node_degree = (
                    self.source_node_degree
                    if domain == "source"
                    else self.target_node_degree
                )
                energy = self.product_geometry.calibrate_structural_energy(
                    raw_energy,
                    node_degree,
                    normalize_by_degree=self.normalize_energy_by_degree,
                    normalize_per_node=self.normalize_energy_per_node,
                )
            topology = (
                self.source_topology_features
                if domain == "source"
                else self.target_topology_features
            )
            gate_module = self.source_gate if domain == "source" else self.target_gate
            gates, _, _ = gate_module(topology, adjacency, energy)
            user_points, item_points = torch.split(
                points, [self.total_num_users, self.total_num_items]
            )
            user_gates, item_gates = torch.split(
                gates, [self.total_num_users, self.total_num_items]
            )
            return user_points, item_points, user_gates, item_gates

        curvature = self.source_curvature if domain == "source" else self.target_curvature
        points = expmap0(tangent, curvature)
        user_points, item_points = torch.split(
            points, [self.total_num_users, self.total_num_items]
        )
        return user_points, item_points, None, None

    def _base_forward_with_gates(
        self,
    ) -> tuple[
        Tensor, Tensor, Tensor, Tensor,
        Tensor | None, Tensor | None, Tensor | None, Tensor | None,
    ]:
        source_users, source_items, source_user_gates, source_item_gates = (
            self._encode_domain(
                self.source_user_embedding,
                self.source_item_embedding,
                self.source_encoder,
                self.source_adjacency,
                "source",
            )
        )
        target_users, target_items, target_user_gates, target_item_gates = (
            self._encode_domain(
                self.target_user_embedding,
                self.target_item_embedding,
                self.target_encoder,
                self.target_adjacency,
                "target",
            )
        )
        return (
            source_users, source_items, target_users, target_items,
            source_user_gates, source_item_gates, target_user_gates, target_item_gates,
        )

    def _logmap_domain(self, points: Tensor, domain: str) -> Tensor:
        if self.uses_product_manifold:
            spherical_curvature, hyperbolic_curvature = self._domain_curvatures(domain)
            return self.product_geometry.logmap0(
                points, spherical_curvature, hyperbolic_curvature
            )
        curvature = (
            self.source_curvature if domain == "source" else self.target_curvature
        )
        return logmap0(points, curvature)

    def _distribution_coordinates(
        self, points: Tensor, gates: Tensor | None, domain: str
    ) -> Tensor:
        tangent = self._logmap_domain(points, domain)
        if not self.uses_product_manifold:
            return tangent / math.sqrt(self.embedding_size)
        while gates is not None and gates.ndim < tangent.ndim:
            gates = gates.unsqueeze(-2)
        factors = torch.split(tangent, self.product_geometry.dimensions, dim=-1)
        return torch.cat([
            factor / math.sqrt(dimension)
            * (1.0 if gates is None else gates[..., index:index + 1].clamp_min(1e-12).sqrt())
            for index, (factor, dimension) in enumerate(
                zip(factors, self.product_geometry.dimensions)
            )
        ], dim=-1)

    def _density_energy_arguments(self, domain: str) -> dict[str, Tensor]:
        if self.uses_product_manifold:
            spherical, hyperbolic = self._domain_curvatures(domain)
            if self.detach_how2_auxiliary_base:
                spherical, hyperbolic = spherical.detach(), hyperbolic.detach()
            return {
                "spherical_curvature": spherical,
                "hyperbolic_curvature": hyperbolic,
            }
        curvature = self.source_curvature if domain == "source" else self.target_curvature
        if self.detach_how2_auxiliary_base:
            curvature = curvature.detach()
        return {"hyperbolic_curvature": curvature}

    def _expmap_domain(self, tangent: Tensor, domain: str) -> Tensor:
        if self.uses_product_manifold:
            spherical_curvature, hyperbolic_curvature = self._domain_curvatures(domain)
            return self.product_geometry.expmap0(
                tangent, spherical_curvature, hyperbolic_curvature
            )
        curvature = (
            self.source_curvature if domain == "source" else self.target_curvature
        )
        return expmap0(tangent, curvature)

    def _expmap_auxiliary_domain(self, tangent: Tensor, domain: str) -> Tensor:
        """Map auxiliary views without updating How1 curvature parameters."""

        if self.uses_product_manifold:
            spherical_curvature, hyperbolic_curvature = self._domain_curvatures(domain)
            if self.detach_how2_auxiliary_base:
                spherical_curvature = spherical_curvature.detach()
                hyperbolic_curvature = hyperbolic_curvature.detach()
            return self.product_geometry.expmap0(
                tangent, spherical_curvature, hyperbolic_curvature
            )
        curvature = (
            self.source_curvature if domain == "source" else self.target_curvature
        )
        if self.detach_how2_auxiliary_base:
            curvature = curvature.detach()
        return expmap0(tangent, curvature)

    def _representations(self) -> ModelRepresentations:
        base = self._base_forward_with_gates()
        if not self.enable_how2:
            return ModelRepresentations(
                *base, None, None, base[0], base[2]
            )
        source_tangent = self._logmap_domain(base[0], "source")
        target_tangent = self._logmap_domain(base[2], "target")
        source_streams = self.disentangler.from_tangent(
            source_tangent,
            lambda value: self._expmap_domain(value, "source"),
            "source",
            residual_scale=self.how2_adaptive_scale,
            detach_auxiliary_base=self.detach_how2_auxiliary_base,
        )
        target_streams = self.disentangler.from_tangent(
            target_tangent,
            lambda value: self._expmap_domain(value, "target"),
            "target",
            residual_scale=self.how2_adaptive_scale,
            detach_auxiliary_base=self.detach_how2_auxiliary_base,
        )
        return ModelRepresentations(
            source_streams.fused, base[1], target_streams.fused, base[3],
            base[4], base[5], base[6], base[7],
            source_streams, target_streams, base[0], base[2],
        )

    def preference_streams(self) -> tuple[DualStreamOutput, DualStreamOutput]:
        representations = self._representations()
        if (
            representations.source_streams is None
            or representations.target_streams is None
        ):
            raise RuntimeError("How2 is disabled")
        return representations.source_streams, representations.target_streams

    def _forward_with_gates(self):
        return self._representations().encoded()

    def forward(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        encoded = self._forward_with_gates()
        return encoded[0], encoded[1], encoded[2], encoded[3]

    def _encoded_for_inference(self):
        """Reuse graph encodings across full-sort batches while the model is in eval mode."""

        if self.training:
            return self._forward_with_gates()
        if self._evaluation_cache is None:
            self._evaluation_cache = tuple(
                value.detach() if value is not None else None
                for value in self._forward_with_gates()
            )
        return self._evaluation_cache


    # Ranking and manifold distance

    def _squared_metric(
        self,
        x: Tensor,
        y: Tensor,
        x_weights: Tensor | None,
        y_weights: Tensor | None,
        domain: str,
    ) -> Tensor:
        if not self.uses_product_manifold:
            curvature = self.source_curvature if domain == "source" else self.target_curvature
            return squared_distance(x, y, curvature)
        if x_weights is None or y_weights is None:
            raise ValueError("product-manifold distance requires node gates")
        spherical_curvature, hyperbolic_curvature = self._domain_curvatures(domain)
        weights = 0.5 * (x_weights + y_weights)
        return self.product_geometry.squared_distance(
            x, y, weights, spherical_curvature, hyperbolic_curvature
        )

    def _pairwise_squared_metric(
        self,
        x: Tensor,
        y: Tensor,
        x_weights: Tensor | None,
        y_weights: Tensor | None,
        domain: str,
    ) -> Tensor:
        if not self.uses_product_manifold:
            curvature = self.source_curvature if domain == "source" else self.target_curvature
            return pairwise_squared_distance(x, y, curvature)
        if x_weights is None or y_weights is None:
            raise ValueError("product-manifold distance requires node gates")
        spherical_curvature, hyperbolic_curvature = self._domain_curvatures(domain)
        return self.product_geometry.pairwise_squared_distance(
            x, y, x_weights, y_weights, spherical_curvature, hyperbolic_curvature
        )

    def _domain_ranking_loss(
        self,
        users: Tensor,
        positive_items: Tensor,
        negative_items: Tensor,
        user_points: Tensor,
        item_points: Tensor,
        user_weights: Tensor | None,
        item_weights: Tensor | None,
        domain: str,
        user_table: nn.Embedding,
        item_table: nn.Embedding,
    ) -> Tensor:
        user = user_points[users]
        positive = item_points[positive_items]
        negative = item_points[negative_items]
        selected_user_weights = None if user_weights is None else user_weights[users]
        positive_weights = None if item_weights is None else item_weights[positive_items]
        negative_weights = None if item_weights is None else item_weights[negative_items]
        if self.enable_how2:
            interests, mixture, _ = self._multi_interest_bundle(
                user, selected_user_weights, domain
            )
            positive_score = self._multi_interest_pair_scores(
                interests, mixture, positive, selected_user_weights,
                positive_weights, domain,
            )
            negative_score = self._multi_interest_pair_scores(
                interests, mixture, negative, selected_user_weights,
                negative_weights, domain,
            )
        else:
            positive_score = -self._squared_metric(
                user, positive, selected_user_weights, positive_weights, domain
            )
            negative_score = -self._squared_metric(
                user, negative, selected_user_weights, negative_weights, domain
            )
        if self.loss_type == "bpr":
            ranking = -F.logsigmoid(positive_score - negative_score).mean()
        else:
            ranking = F.relu(
                negative_score - positive_score + self.margin
            ).mean()
        regularization = (
            user_table(users).square().sum(dim=-1).mean()
            + item_table(positive_items).square().sum(dim=-1).mean()
            + item_table(negative_items).square().sum(dim=-1).mean()
        )
        return ranking + self.reg_weight * regularization


    # Cross-domain contrastive transfer

    def _sample_negative_items(self, positives: Tensor, domain: str) -> Tensor:
        catalog = (
            self.source_item_catalog if domain == "source" else self.target_item_catalog
        )
        catalog = catalog[catalog > 0]
        if catalog.numel() < 2:
            raise ValueError(f"{domain} item catalog needs at least two real items")
        shape = (positives.numel(), self.num_transfer_negatives)
        indices = torch.randint(catalog.numel(), shape, device=positives.device)
        negatives = catalog[indices]
        equal = negatives.eq(positives.reshape(-1, 1))
        if equal.any():
            negatives[equal] = catalog[(indices[equal] + 1) % catalog.numel()]
        return negatives

    def _overlap_neighbors(self, batch_users: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        overlap = torch.unique(batch_users)
        overlap = overlap[(overlap > 0) & (overlap < self.overlapped_num_users)]
        if self.max_transfer_users is not None and overlap.numel() > self.max_transfer_users:
            selection = torch.randperm(overlap.numel(), device=overlap.device)[
                : self.max_transfer_users
            ]
            overlap = overlap[selection]

        source_start = self.source_neighbor_offsets[overlap]
        source_stop = self.source_neighbor_offsets[overlap + 1]
        target_start = self.target_neighbor_offsets[overlap]
        target_stop = self.target_neighbor_offsets[overlap + 1]
        source_length = source_stop - source_start
        target_length = target_stop - target_start
        valid = (source_length > 0) & (target_length > 0)
        users = overlap[valid]
        source_start, source_length = source_start[valid], source_length[valid]
        target_start, target_length = target_start[valid], target_length[valid]
        if users.numel() == 0:
            empty = batch_users.new_empty(0)
            return empty, empty, empty

        source_offset = (
            torch.rand(users.numel(), device=users.device) * source_length
        ).long()
        target_offset = (
            torch.rand(users.numel(), device=users.device) * target_length
        ).long()
        source_items = self.source_neighbor_items[source_start + source_offset]
        target_items = self.target_neighbor_items[target_start + target_offset]
        return users, source_items, target_items

    def _diagonal_nce(
        self,
        anchors: Tensor,
        positives: Tensor,
        anchor_weights: Tensor | None,
        positive_weights: Tensor | None,
        domain: str,
    ) -> Tensor:
        if anchors.size(0) < 2:
            return anchors.sum() * 0.0
        total = anchors.new_zeros(())
        count = anchors.size(0)
        for start in range(0, count, self.contrastive_chunk_size):
            stop = min(start + self.contrastive_chunk_size, count)
            chunk_weights = (
                None if anchor_weights is None else anchor_weights[start:stop]
            )
            logits = -self._pairwise_squared_metric(
                anchors[start:stop],
                positives,
                chunk_weights,
                positive_weights,
                domain,
            ) / self.temperature
            labels = torch.arange(start, stop, device=anchors.device)
            total = total + F.cross_entropy(logits, labels, reduction="sum")
        return total / count

    def _sampled_nce(
        self,
        anchors: Tensor,
        positives: Tensor,
        negative_points: Tensor,
        anchor_weights: Tensor | None,
        positive_weights: Tensor | None,
        negative_weights: Tensor | None,
        domain: str,
    ) -> Tensor:
        positive_logits = -self._squared_metric(
            anchors, positives, anchor_weights, positive_weights, domain
        ).unsqueeze(1)
        expanded_anchor_weights = (
            None if anchor_weights is None else anchor_weights.unsqueeze(1)
        )
        negative_logits = -self._squared_metric(
            anchors.unsqueeze(1),
            negative_points,
            expanded_anchor_weights,
            negative_weights,
            domain,
        )
        logits = torch.cat([positive_logits, negative_logits], dim=1) / self.temperature
        labels = torch.zeros(anchors.size(0), dtype=torch.long, device=anchors.device)
        return F.cross_entropy(logits, labels)

    def _map_between_domains(
        self,
        mapper: nn.Module,
        points: Tensor,
        from_domain: str,
        to_domain: str,
    ) -> Tensor:
        if not self.uses_product_manifold:
            c_from = (
                self.source_curvature if from_domain == "source" else self.target_curvature
            )
            c_to = self.source_curvature if to_domain == "source" else self.target_curvature
            return mapper(points, c_from, c_to)
        source_spherical, source_hyperbolic = self._domain_curvatures(from_domain)
        target_spherical, target_hyperbolic = self._domain_curvatures(to_domain)
        return mapper(
            points,
            source_spherical,
            source_hyperbolic,
            target_spherical,
            target_hyperbolic,
        )

    def _directional_transfer(
        self,
        *,
        from_users: Tensor,
        from_items: Tensor,
        to_users: Tensor,
        to_items: Tensor,
        from_user_weights: Tensor | None,
        from_item_weights: Tensor | None,
        to_user_weights: Tensor | None,
        to_item_weights: Tensor | None,
        overlap_users: Tensor,
        from_positive_items: Tensor,
        to_positive_items: Tensor,
        mapper: nn.Module,
        from_domain: str,
        to_domain: str,
    ) -> Tensor:
        transfer_users = from_users[overlap_users]
        transfer_items = from_items[from_positive_items]
        mapped_user_weights = (
            None if from_user_weights is None else from_user_weights[overlap_users]
        )
        mapped_item_weights = (
            None if from_item_weights is None else from_item_weights[from_positive_items]
        )
        if self.detach_transferred:
            transfer_users = transfer_users.detach()
            transfer_items = transfer_items.detach()
            if mapped_user_weights is not None:
                mapped_user_weights = mapped_user_weights.detach()
                mapped_item_weights = mapped_item_weights.detach()
        mapped_users = self._map_between_domains(
            mapper, transfer_users, from_domain, to_domain
        )
        mapped_items = self._map_between_domains(
            mapper, transfer_items, from_domain, to_domain
        )
        destination_users = to_users[overlap_users]
        destination_items = to_items[to_positive_items]
        destination_user_weights = (
            None if to_user_weights is None else to_user_weights[overlap_users]
        )
        destination_item_weights = (
            None if to_item_weights is None else to_item_weights[to_positive_items]
        )

        losses: list[Tensor] = []
        if self.enabled_relations["uu"]:
            losses.append(
                self._diagonal_nce(
                    mapped_users, destination_users, mapped_user_weights,
                    destination_user_weights, to_domain,
                )
            )
        if self.enabled_relations["ui"]:
            negative_ids = self._sample_negative_items(to_positive_items, to_domain)
            negative_weights = (
                None if to_item_weights is None else to_item_weights[negative_ids]
            )
            losses.append(
                self._sampled_nce(
                    mapped_users, destination_items, to_items[negative_ids],
                    mapped_user_weights, destination_item_weights, negative_weights,
                    to_domain,
                )
            )
        if self.enabled_relations["iu"]:
            losses.append(
                self._diagonal_nce(
                    mapped_items, destination_users, mapped_item_weights,
                    destination_user_weights, to_domain,
                )
            )
        if self.enabled_relations["ii"]:
            negative_ids = self._sample_negative_items(to_positive_items, to_domain)
            negative_weights = (
                None if to_item_weights is None else to_item_weights[negative_ids]
            )
            losses.append(
                self._sampled_nce(
                    mapped_items, destination_items, to_items[negative_ids],
                    mapped_item_weights, destination_item_weights, negative_weights,
                    to_domain,
                )
            )
        if not losses:
            return mapped_users.sum() * 0.0
        return torch.stack(losses).mean()

    def transfer_loss(
        self,
        source_users: Tensor,
        source_items: Tensor,
        target_users: Tensor,
        target_items: Tensor,
        batch_source_users: Tensor,
        batch_target_users: Tensor,
        source_user_weights: Tensor | None = None,
        source_item_weights: Tensor | None = None,
        target_user_weights: Tensor | None = None,
        target_item_weights: Tensor | None = None,
    ) -> Tensor:
        batch_overlap = torch.cat([batch_source_users, batch_target_users])
        users, source_positive, target_positive = self._overlap_neighbors(batch_overlap)
        if users.numel() == 0:
            return sum(
                parameter.sum() * 0.0
                for parameter in self.source_to_target.parameters()
            )

        source_to_target = self._directional_transfer(
            from_users=source_users,
            from_items=source_items,
            to_users=target_users,
            to_items=target_items,
            from_user_weights=source_user_weights,
            from_item_weights=source_item_weights,
            to_user_weights=target_user_weights,
            to_item_weights=target_item_weights,
            overlap_users=users,
            from_positive_items=source_positive,
            to_positive_items=target_positive,
            mapper=self.source_to_target,
            from_domain="source",
            to_domain="target",
        )
        if not self.bidirectional:
            return source_to_target
        target_to_source = self._directional_transfer(
            from_users=target_users,
            from_items=target_items,
            to_users=source_users,
            to_items=source_items,
            from_user_weights=target_user_weights,
            from_item_weights=target_item_weights,
            to_user_weights=source_user_weights,
            to_item_weights=source_item_weights,
            overlap_users=users,
            from_positive_items=target_positive,
            to_positive_items=source_positive,
            mapper=self.target_to_source,
            from_domain="target",
            to_domain="source",
        )
        return 0.5 * (source_to_target + target_to_source)


    # HOW2 density and multi-interest learning

    def _valid_unique_users(self, users: Tensor, upper_bound: int) -> Tensor:
        users = torch.unique(users.long())
        return users[(users > 0) & (users < upper_bound)]

    def _cluster_loss_and_assignments(
        self,
        clusterer: EntropyManifoldClusterer,
        points: Tensor,
        point_gates: Tensor | None,
        domain: str,
    ) -> tuple[Tensor, Tensor]:
        if not self.uses_product_manifold:
            curvature = (
                self.source_curvature if domain == "source" else self.target_curvature
            )
            if self.detach_how2_auxiliary_base:
                curvature = curvature.detach()
            return clusterer(points, curvature)
        if point_gates is None:
            raise ValueError("product-manifold clustering requires node gates")
        spherical_curvature, hyperbolic_curvature = self._domain_curvatures(domain)
        if self.detach_how2_auxiliary_base:
            spherical_curvature = spherical_curvature.detach()
            hyperbolic_curvature = hyperbolic_curvature.detach()
        centers = self.product_geometry.expmap0(
            clusterer.centers_tangent, spherical_curvature, hyperbolic_curvature
        )
        gate_module = self.source_gate if domain == "source" else self.target_gate
        center_gates = gate_module.component_prior.to(
            dtype=point_gates.dtype, device=point_gates.device
        ).expand(clusterer.num_clusters, -1)
        distances = self.product_geometry.pairwise_squared_distance(
            points, centers, point_gates, center_gates,
            spherical_curvature, hyperbolic_curvature,
        )
        return clusterer.loss_from_distances(distances)

    def _cluster_assignments(
        self,
        clusterer: EntropyManifoldClusterer,
        points: Tensor,
        point_gates: Tensor | None,
        domain: str,
    ) -> Tensor:
        if not self.uses_product_manifold:
            curvature = (
                self.source_curvature if domain == "source" else self.target_curvature
            )
            return clusterer.assignments(points, curvature)[0]
        return self._cluster_loss_and_assignments(
            clusterer, points, point_gates, domain
        )[1]

    def _multi_interest_bundle(
        self,
        user_points: Tensor,
        user_gates: Tensor | None,
        domain: str,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not self.enable_how2 or self.shared_clusterer is None:
            raise RuntimeError("multi-interest preferences require How2")
        mixture_weights = self._cluster_assignments(
            self.shared_clusterer, user_points, user_gates, domain
        )
        base_tangent = self._logmap_domain(user_points, domain)
        interest_tangent = base_tangent.unsqueeze(1) + self.how2_adaptive_scale * (
            self.shared_clusterer.centers_tangent.unsqueeze(0)
        )
        interest_points = self._expmap_domain(interest_tangent, domain)
        return interest_points, mixture_weights, interest_tangent

    def _multi_interest_pair_scores(
        self,
        interest_points: Tensor,
        mixture_weights: Tensor,
        item_points: Tensor,
        user_gates: Tensor | None,
        item_gates: Tensor | None,
        domain: str,
    ) -> Tensor:
        expanded_user_gates = None if user_gates is None else user_gates.unsqueeze(1)
        expanded_item_gates = None if item_gates is None else item_gates.unsqueeze(1)
        distances = self._squared_metric(
            interest_points,
            item_points.unsqueeze(1),
            expanded_user_gates,
            expanded_item_gates,
            domain,
        )
        log_mixture = mixture_weights.clamp_min(1e-12).log()
        temperature = self.multi_interest_temperature
        return temperature * torch.logsumexp(
            log_mixture - distances / temperature, dim=-1
        )

    def _multi_interest_full_scores(
        self,
        interest_points: Tensor,
        mixture_weights: Tensor,
        item_points: Tensor,
        user_gates: Tensor | None,
        item_gates: Tensor | None,
        domain: str,
    ) -> Tensor:
        batch_size, num_interests = interest_points.shape[:2]
        flat_interests = interest_points.reshape(
            batch_size * num_interests, interest_points.size(-1)
        )
        flat_gates = (
            None
            if user_gates is None
            else user_gates.unsqueeze(1)
            .expand(-1, num_interests, -1)
            .reshape(batch_size * num_interests, -1)
        )
        log_mixture = mixture_weights.clamp_min(1e-12).log().unsqueeze(-1)
        temperature = self.multi_interest_temperature
        score_chunks = []
        for start in range(0, item_points.size(0), self.full_sort_item_chunk_size):
            stop = min(start + self.full_sort_item_chunk_size, item_points.size(0))
            chunk_gates = None if item_gates is None else item_gates[start:stop]
            distances = self._pairwise_squared_metric(
                flat_interests,
                item_points[start:stop],
                flat_gates,
                chunk_gates,
                domain,
            ).reshape(batch_size, num_interests, stop - start)
            score_chunks.append(
                temperature
                * torch.logsumexp(log_mixture - distances / temperature, dim=1)
            )
        return torch.cat(score_chunks, dim=1)

    def _how2_losses(
        self,
        representations: ModelRepresentations,
        batch_source_users: Tensor,
        batch_target_users: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        source_streams = representations.source_streams
        target_streams = representations.target_streams
        if (
            source_streams is None
            or target_streams is None
            or self.how2_loss_scale <= 0.0
        ):
            zero = representations.source_users.sum() * 0.0
            return zero, zero, zero

        source_ids = self._valid_unique_users(
            batch_source_users, self.total_num_users
        )
        target_ids = self._valid_unique_users(
            batch_target_users, self.target_num_users
        )
        source_shared_tangent = (
            source_streams.shared_auxiliary_tangent[source_ids]
        )
        source_private_tangent = (
            source_streams.private_auxiliary_tangent[source_ids]
        )
        target_shared_tangent = (
            target_streams.shared_auxiliary_tangent[target_ids]
        )
        target_private_tangent = (
            target_streams.private_auxiliary_tangent[target_ids]
        )

        disentanglement = 0.5 * (
            self.wasserstein_independence(
                source_shared_tangent, source_private_tangent
            )
            + self.wasserstein_independence(
                target_shared_tangent, target_private_tangent
            )
        )

        loss_arguments = {
            "temperature": self.density_energy_temperature,
            "margin": self.density_energy_margin,
            "noise_scale": self.density_energy_noise_scale,
            "energy_l2": self.density_energy_l2,
        }
        source_geometry = self._density_energy_arguments("source")
        target_geometry = self._density_energy_arguments("target")
        source_shared_energy = self.shared_density_energy.contrastive_loss(
            source_shared_tangent, **loss_arguments, **source_geometry
        )
        target_shared_energy = self.shared_density_energy.contrastive_loss(
            target_shared_tangent, **loss_arguments, **target_geometry
        )
        source_private_energy = self.source_private_density_energy.contrastive_loss(
            source_private_tangent, **loss_arguments, **source_geometry
        )
        target_private_energy = self.target_private_density_energy.contrastive_loss(
            target_private_tangent, **loss_arguments, **target_geometry
        )
        density_energy = 0.5 * (
            source_shared_energy + target_shared_energy
            + source_private_energy + target_private_energy
        )

        overlap_ids = torch.unique(torch.cat([source_ids, target_ids]))
        overlap_ids = overlap_ids[
            (overlap_ids > 0) & (overlap_ids < self.overlapped_num_users)
        ]
        if overlap_ids.numel() > 0:
            source_overlap_energy = self.shared_density_energy.from_tangent(
                source_streams.shared_auxiliary_tangent[overlap_ids],
                **self._density_energy_arguments("source"),
            )
            target_overlap_energy = self.shared_density_energy.from_tangent(
                target_streams.shared_auxiliary_tangent[overlap_ids],
                **self._density_energy_arguments("target"),
            )
            density_energy = density_energy + F.smooth_l1_loss(
                source_overlap_energy, target_overlap_energy
            )

        source_gates = representations.source_user_gates
        target_gates = representations.target_user_gates
        if self.detach_how2_auxiliary_base:
            source_gates = None if source_gates is None else source_gates.detach()
            target_gates = None if target_gates is None else target_gates.detach()
        source_shared_points = self._expmap_auxiliary_domain(
            source_streams.shared_auxiliary_tangent, "source"
        )
        target_shared_points = self._expmap_auxiliary_domain(
            target_streams.shared_auxiliary_tangent, "target"
        )
        source_private_points = self._expmap_auxiliary_domain(
            source_streams.private_auxiliary_tangent, "source"
        )
        target_private_points = self._expmap_auxiliary_domain(
            target_streams.private_auxiliary_tangent, "target"
        )
        source_shared_cluster, _ = self._cluster_loss_and_assignments(
            self.shared_clusterer, source_shared_points[source_ids],
            None if source_gates is None else source_gates[source_ids], "source",
        )
        target_shared_cluster, _ = self._cluster_loss_and_assignments(
            self.shared_clusterer, target_shared_points[target_ids],
            None if target_gates is None else target_gates[target_ids], "target",
        )
        source_private_cluster, _ = self._cluster_loss_and_assignments(
            self.source_private_clusterer, source_private_points[source_ids],
            None if source_gates is None else source_gates[source_ids], "source",
        )
        target_private_cluster, _ = self._cluster_loss_and_assignments(
            self.target_private_clusterer, target_private_points[target_ids],
            None if target_gates is None else target_gates[target_ids], "target",
        )
        clustering = 0.25 * (
            source_shared_cluster + target_shared_cluster
            + source_private_cluster + target_private_cluster
        )
        clustering = clustering + self.interest_separation_weight * (
            self.shared_clusterer.separation_loss(self.interest_separation_margin)
        )
        if overlap_ids.numel() > 0:
            source_assignments = self._cluster_assignments(
                self.shared_clusterer, source_shared_points[overlap_ids],
                None if source_gates is None else source_gates[overlap_ids],
                "source",
            )
            target_assignments = self._cluster_assignments(
                self.shared_clusterer, target_shared_points[overlap_ids],
                None if target_gates is None else target_gates[overlap_ids],
                "target",
            )
            clustering = clustering + symmetrized_assignment_kl(
                source_assignments, target_assignments
            )
        scale = self.how2_loss_scale
        return (
            scale * disentanglement,
            scale * density_energy,
            scale * clustering,
        )


    # HOW3 selective alignment

    def _how3_overlap_profiles(
        self, batch_users: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        return self._overlap_neighbors(batch_users)

    def _how3_components(
        self,
        representations: ModelRepresentations,
        users: Tensor,
        source_items: Tensor,
        target_items: Tensor,
        *,
        destination_users: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        if representations.source_streams is None or representations.target_streams is None:
            raise RuntimeError("How3 requires How2 shared streams")
        destination_users = users if destination_users is None else destination_users
        source_users = representations.source_streams.shared[users]
        target_users = representations.target_streams.shared[destination_users]
        source_item_points = representations.source_items[source_items]
        target_item_points = representations.target_items[target_items]
        source_user_gates = (
            None if representations.source_user_gates is None
            else representations.source_user_gates[users]
        )
        target_user_gates = (
            None if representations.target_user_gates is None
            else representations.target_user_gates[destination_users]
        )
        source_item_gates = (
            None if representations.source_item_gates is None
            else representations.source_item_gates[source_items]
        )
        target_item_gates = (
            None if representations.target_item_gates is None
            else representations.target_item_gates[target_items]
        )
        source_interests, source_mixture, source_tangent = self._multi_interest_bundle(
            source_users, source_user_gates, "source"
        )
        target_interests, target_mixture, target_tangent = self._multi_interest_bundle(
            target_users, target_user_gates, "target"
        )
        transfer_users = (
            source_interests.detach() if self.align_detach_source else source_interests
        )
        transfer_items = (
            source_item_points.detach() if self.align_detach_source else source_item_points
        )
        mapped_users = self._map_between_domains(
            self.source_to_target, transfer_users, "source", "target"
        )
        mapped_items = self._map_between_domains(
            self.source_to_target, transfer_items, "source", "target"
        )
        source_bundle_gates = (
            None if source_user_gates is None else source_user_gates.unsqueeze(1)
        )
        target_bundle_gates = (
            None if target_user_gates is None else target_user_gates.unsqueeze(1)
        )
        interest_w2 = self._squared_metric(
            mapped_users, target_interests,
            source_bundle_gates, target_bundle_gates, "target",
        )
        bundle_mixture = 0.5 * (source_mixture + target_mixture)
        bundle_mixture = bundle_mixture / bundle_mixture.sum(dim=-1, keepdim=True)
        user_w2 = (bundle_mixture * interest_w2).sum(dim=-1)
        item_w2 = self._squared_metric(
            mapped_items, target_item_points, source_item_gates, target_item_gates, "target"
        )
        sample_w2 = 0.5 * (user_w2 + item_w2)
        with torch.no_grad():
            source_density = self.shared_density_energy.from_tangent(
                source_tangent, **self._density_energy_arguments("source")
            )
            target_density = self.shared_density_energy.from_tangent(
                target_tangent, **self._density_energy_arguments("target")
            )
            density = 0.5 * (
                (source_mixture * source_density).sum(dim=-1)
                + (target_mixture * target_density).sum(dim=-1)
            )
            roundtrip = self._map_between_domains(
                self.target_to_source, mapped_users.detach(), "target", "source"
            )
            roundtrip_distance = self._squared_metric(
                roundtrip, source_interests.detach(),
                source_bundle_gates, source_bundle_gates, "source",
            )
            cycle_consistency = (source_mixture * roundtrip_distance).sum(dim=-1)
            source_relation = self._squared_metric(
                source_interests, source_item_points.unsqueeze(1),
                source_bundle_gates,
                None if source_item_gates is None else source_item_gates.unsqueeze(1),
                "source",
            )
            target_relation = self._squared_metric(
                target_interests, target_item_points.unsqueeze(1),
                target_bundle_gates,
                None if target_item_gates is None else target_item_gates.unsqueeze(1),
                "target",
            )
            mapped_relation = self._squared_metric(
                mapped_users, mapped_items.unsqueeze(1),
                source_bundle_gates,
                None if source_item_gates is None else source_item_gates.unsqueeze(1),
                "target",
            )
            source_relation = (source_mixture * source_relation).sum(dim=-1)
            target_relation = (target_mixture * target_relation).sum(dim=-1)
            mapped_relation = (source_mixture * mapped_relation).sum(dim=-1)
            behavioral_consistency = (source_relation - target_relation).abs()
            isometry_consistency = (source_relation - mapped_relation).abs()
            midpoint = 0.5 * (source_mixture + target_mixture)
            mixture_consistency = 0.5 * (
                (source_mixture * (
                    source_mixture.clamp_min(1e-12).log()
                    - midpoint.clamp_min(1e-12).log()
                )).sum(dim=-1)
                + (target_mixture * (
                    target_mixture.clamp_min(1e-12).log()
                    - midpoint.clamp_min(1e-12).log()
                )).sum(dim=-1)
            )
            consistency_terms = torch.stack([
                cycle_consistency,
                behavioral_consistency,
                isometry_consistency,
                mixture_consistency,
            ])
            scales = consistency_terms.mean(dim=1, keepdim=True).clamp_min(1e-8)
            cross = (consistency_terms / scales).mean(dim=0)
        return (
            density, cross, sample_w2, mapped_users, mapped_items,
            target_interests, target_item_points, source_user_gates, target_user_gates,
            bundle_mixture,
        )

    def _weighted_bundle_diagonal_nce(
        self,
        anchors: Tensor,
        positives: Tensor,
        mixture: Tensor,
        weights: Tensor,
        anchor_gates: Tensor | None,
        positive_gates: Tensor | None,
    ) -> Tensor:
        if anchors.size(0) < 2:
            return anchors.sum() * 0.0
        component_logits = []
        for index in range(anchors.size(1)):
            component_logits.append(
                -self._pairwise_squared_metric(
                    anchors[:, index],
                    positives[:, index],
                    anchor_gates,
                    positive_gates,
                    "target",
                ) / self.multi_interest_temperature
            )
        logits = torch.stack(component_logits, dim=1)
        logits = (
            self.multi_interest_temperature
            * torch.logsumexp(
                mixture.clamp_min(1e-12).log().unsqueeze(-1) + logits,
                dim=1,
            )
            / self.temperature
        )
        labels = torch.arange(anchors.size(0), device=anchors.device)
        per_sample = F.cross_entropy(logits, labels, reduction="none")
        return (weights * per_sample).sum()

    def _weighted_diagonal_nce(
        self,
        anchors: Tensor,
        positives: Tensor,
        weights: Tensor,
        anchor_gates: Tensor | None,
        positive_gates: Tensor | None,
    ) -> Tensor:
        if anchors.size(0) < 2:
            return anchors.sum() * 0.0
        logits = -self._pairwise_squared_metric(
            anchors, positives, anchor_gates, positive_gates, "target"
        ) / self.temperature
        labels = torch.arange(anchors.size(0), device=anchors.device)
        per_sample = F.cross_entropy(logits, labels, reduction="none")
        return (weights * per_sample).sum()

    def selective_alignment_loss(
        self, representations: ModelRepresentations, batch_users: Tensor
    ) -> Tensor:
        zero = representations.source_users.sum() * 0.0
        if not self.how3_active or self.how3 is None or not self.how3.is_calibrated:
            return zero
        users, source_items, target_items = self._how3_overlap_profiles(batch_users)
        if users.numel() == 0:
            return zero
        (
            density, cross, _sample_w2, mapped_users, mapped_items,
            target_users, target_item_points, source_user_gates, target_user_gates,
            bundle_mixture,
        ) = self._how3_components(representations, users, source_items, target_items)
        energy = self.how3.score(density, cross)
        weights, _ = self.how3.trusted_weights(energy)
        mapped_user_coordinates = self._distribution_coordinates(
            mapped_users, source_user_gates, "target"
        ).reshape(-1, self.embedding_size)
        target_user_coordinates = self._distribution_coordinates(
            target_users, target_user_gates, "target"
        ).reshape(-1, self.embedding_size)
        user_atom_weights = (weights.unsqueeze(1) * bundle_mixture).reshape(-1)
        source_item_gates = (
            None if representations.source_item_gates is None
            else representations.source_item_gates[source_items]
        )
        target_item_gates = (
            None if representations.target_item_gates is None
            else representations.target_item_gates[target_items]
        )
        user_distribution = self.how3.alignment(
            mapped_user_coordinates,
            target_user_coordinates,
            user_atom_weights,
            user_atom_weights,
            num_projections=self.distribution_num_projections,
        )
        item_distribution = self.how3.alignment(
            self._distribution_coordinates(mapped_items, source_item_gates, "target"),
            self._distribution_coordinates(target_item_points, target_item_gates, "target"),
            weights,
            weights,
            num_projections=self.distribution_num_projections,
        )
        align = 0.5 * (user_distribution + item_distribution)
        if self.selective_contrastive_weight > 0:
            contrastive = 0.5 * (
                self._weighted_bundle_diagonal_nce(
                    mapped_users, target_users, bundle_mixture, weights,
                    source_user_gates, target_user_gates,
                )
                + self._weighted_diagonal_nce(
                    mapped_items, target_item_points, weights,
                    source_item_gates, target_item_gates,
                )
            )
            align = align + self.selective_contrastive_weight * contrastive
        return align

    @torch.no_grad()
    def fit_how3_threshold(self, validation_users: Tensor) -> float:
        if self.how3 is None:
            raise RuntimeError("How3 is disabled")
        was_training = self.training
        self.eval()
        try:
            representations = self._representations()
            users, source_items, target_items = self._how3_overlap_profiles(validation_users)
            if users.numel() == 0:
                return float(self.how3.calibrator.threshold)
            density, cross, *_ = self._how3_components(
                representations, users, source_items, target_items
            )
            if users.numel() > 1:
                null_users = torch.roll(users, shifts=1)
                null_density, null_cross, *_ = self._how3_components(
                    representations, users, source_items, target_items,
                    destination_users=null_users,
                )
            else:
                null_density = null_cross = None
            return self.how3.fit(
                density, cross, null_density=null_density, null_cross=null_cross
            )
        finally:
            self.train(was_training)

    def split_how123_loss(self, breakdown: LossBreakdown) -> How123Losses:
        raw_align = self.effective_selective_align_weight * breakdown.selective_align
        primary = breakdown.total - raw_align
        align = raw_align
        if self.align_loss_cap_ratio > 0:
            scale = (
                self.align_loss_cap_ratio
                * primary.detach().abs().clamp_min(1e-12)
                / raw_align.detach().abs().clamp_min(1e-12)
            ).clamp(max=1.0)
            align = raw_align * scale
        return How123Losses(primary=primary, align=align, breakdown=breakdown)

    def private_parameters(self) -> list[nn.Parameter]:
        if not self.enable_how2:
            return []
        return [
            *self.disentangler.private_branches.parameters(),
            *self.source_private_density_energy.parameters(),
            *self.target_private_density_energy.parameters(),
            *self.source_private_clusterer.parameters(),
            *self.target_private_clusterer.parameters(),
        ]

    def conflict_parameters(self) -> tuple[list[str], list[nn.Parameter]]:
        private_ids = {id(parameter) for parameter in self.private_parameters()}
        names, parameters = [], []
        for name, parameter in self.named_parameters():
            if parameter.requires_grad and id(parameter) not in private_ids:
                names.append(name)
                parameters.append(parameter)
        return names, parameters

    def assert_private_isolation(self, overlap_users: Tensor) -> bool:
        private = [parameter for parameter in self.private_parameters() if parameter.requires_grad]
        if not private or self.how3 is None:
            return True
        representations = self._representations()
        align = self.selective_alignment_loss(representations, overlap_users)
        gradients = torch.autograd.grad(align, private, allow_unused=True)
        leaked = any(
            gradient is not None and bool(gradient.detach().abs().max() > 0)
            for gradient in gradients
        )
        self.how3.diagnostics.isolation_ok = not leaked
        if leaked:
            raise RuntimeError("How3 alignment gradient leaked into a private branch")
        return True

    def how3_diagnostics(self) -> dict[str, object]:
        if self.how3 is None:
            return {"how3_enabled": False}
        payload = self.how3.diagnostics.as_dict()
        payload.update({
            "how3_enabled": self.enable_how3,
            "how3_active": self.how3_active,
            "how3_start_epoch": self.how3_start_epoch,
            "how3_epoch": self._training_epoch,
            "how3_loss_scale": self.how3_loss_scale,
            "threshold_calibrated": self.how3.is_calibrated,
            "how3_consumes_how2_density": True,
            "normalized_riemannian_density": True,
            "transfer_score_alignment_independent": True,
            "distribution_level_soft_alignment": True,
            "user_level_bundle_selection": True,
            "interest_component_selection": False,
            "num_interests": self.shared_clusterer.num_clusters,
            "private_hard_isolation": True,
            "product_metric_alignment": self.uses_product_manifold,
        })
        return payload


    # Training objectives

    def _hierarchy_loss(
        self,
        source_users: Tensor,
        source_items: Tensor,
        target_users: Tensor,
        target_items: Tensor,
    ) -> Tensor:
        source_active = torch.cat(
            [
                source_users[self.source_active_users],
                source_items[self.source_active_items],
            ]
        )
        target_active = torch.cat(
            [
                target_users[self.target_active_users],
                target_items[self.target_active_items],
            ]
        )
        if self.uses_product_manifold:
            source_tangent = self.product_geometry.logmap0(
                source_active, self.source_spherical_curvature, self.source_curvature
            )
            target_tangent = self.product_geometry.logmap0(
                target_active, self.target_spherical_curvature, self.target_curvature
            )
        else:
            source_tangent = logmap0(source_active, self.source_curvature)
            target_tangent = logmap0(target_active, self.target_curvature)
        source_radius = source_tangent.square().sum(-1).mean()
        target_radius = target_tangent.square().sum(-1).mean()
        return F.relu(self.hierarchy_radius_sq - source_radius) + F.relu(
            self.hierarchy_radius_sq - target_radius
        )

    def loss(
        self,
        *,
        source_users: Tensor,
        source_positive_items: Tensor,
        source_negative_items: Tensor,
        target_users: Tensor,
        target_positive_items: Tensor,
        target_negative_items: Tensor,
    ) -> LossBreakdown:
        representations = self._representations()
        encoded = representations.encoded()
        (
            source_user_points,
            source_item_points,
            target_user_points,
            target_item_points,
            source_user_gates,
            source_item_gates,
            target_user_gates,
            target_item_gates,
        ) = encoded
        source_loss = self._domain_ranking_loss(
            source_users,
            source_positive_items,
            source_negative_items,
            source_user_points,
            source_item_points,
            source_user_gates,
            source_item_gates,
            "source",
            self.source_user_embedding,
            self.source_item_embedding,
        )
        target_loss = self._domain_ranking_loss(
            target_users,
            target_positive_items,
            target_negative_items,
            target_user_points,
            target_item_points,
            target_user_gates,
            target_item_gates,
            "target",
            self.target_user_embedding,
            self.target_item_embedding,
        )
        transfer_source_users = source_user_points
        transfer_target_users = target_user_points
        if representations.source_streams is not None:
            transfer_source_users = representations.source_streams.shared
            transfer_target_users = representations.target_streams.shared
        transfer = self.transfer_loss(
            transfer_source_users,
            source_item_points,
            transfer_target_users,
            target_item_points,
            source_users,
            target_users,
            source_user_gates,
            source_item_gates,
            target_user_gates,
            target_item_gates,
        )
        hierarchy = self._hierarchy_loss(
            representations.source_base_users,
            source_item_points,
            representations.target_base_users,
            target_item_points,
        )
        gate_balance = self._gate_balance_loss(
            source_user_gates,
            source_item_gates,
            target_user_gates,
            target_item_gates,
        )
        disentanglement, density_energy, clustering = self._how2_losses(
            representations, source_users, target_users
        )
        selective_align = self.selective_alignment_loss(
            representations, torch.cat([source_users, target_users])
        )
        total = (
            source_loss
            + target_loss
            + self.transfer_weight * transfer
            + self.hierarchy_weight * hierarchy
            + self.gate_balance_weight * gate_balance
            + self.disentanglement_weight * disentanglement
            + self.density_energy_weight * density_energy
            + self.cluster_weight * clustering
            + self.effective_selective_align_weight * selective_align
        )
        return LossBreakdown(
            total=total, source=source_loss, target=target_loss,
            transfer=transfer, hierarchy=hierarchy, gate_balance=gate_balance,
            disentanglement=disentanglement, density_energy=density_energy,
            clustering=clustering, selective_align=selective_align,
        )

    def single_domain_loss(
        self,
        *,
        users: Tensor,
        positive_items: Tensor,
        negative_items: Tensor,
        domain: str,
    ) -> Tensor:
        """Compute one domain loss for SOURCE or TARGET training phases."""

        if domain == "source":
            user_embedding = self.source_user_embedding
            item_embedding = self.source_item_embedding
            encoder = self.source_encoder
            adjacency = self.source_adjacency
        elif domain == "target":
            user_embedding = self.target_user_embedding
            item_embedding = self.target_item_embedding
            encoder = self.target_encoder
            adjacency = self.target_adjacency
        else:
            raise ValueError("domain must be source or target")
        user_points, item_points, user_gates, item_gates = self._encode_domain(
            user_embedding, item_embedding, encoder, adjacency, domain
        )
        return self._domain_ranking_loss(
            users,
            positive_items,
            negative_items,
            user_points,
            item_points,
            user_gates,
            item_gates,
            domain,
            user_embedding,
            item_embedding,
        )

    def overlap_loss(self, overlap_users: Tensor) -> Tensor:
        """Compute the weighted transfer objective for an explicit OVERLAP phase."""

        representations = self._representations()
        encoded = representations.encoded()
        if self.how3_active and self.how3 is not None and self.how3.is_calibrated:
            return self.effective_selective_align_weight * self.selective_alignment_loss(
                representations, overlap_users
            )
        source_users = (
            representations.source_streams.shared
            if representations.source_streams is not None else encoded[0]
        )
        target_users = (
            representations.target_streams.shared
            if representations.target_streams is not None else encoded[2]
        )
        transfer = self.transfer_loss(
            source_users,
            encoded[1],
            target_users,
            encoded[3],
            overlap_users,
            overlap_users,
            encoded[4],
            encoded[5],
            encoded[6],
            encoded[7],
        )
        return self.transfer_weight * transfer


    # Prediction and diagnostics

    def score(self, users: Tensor, items: Tensor, domain: str = "target") -> Tensor:
        if domain not in {"source", "target"}:
            raise ValueError("domain must be source or target")
        encoded = self._encoded_for_inference()
        if domain == "source":
            user_points, item_points = encoded[0], encoded[1]
            user_gates, item_gates = encoded[4], encoded[5]
        else:
            user_points, item_points = encoded[2], encoded[3]
            user_gates, item_gates = encoded[6], encoded[7]
        selected_user_gates = None if user_gates is None else user_gates[users]
        selected_item_gates = None if item_gates is None else item_gates[items]
        if self.enable_how2:
            interests, mixture, _ = self._multi_interest_bundle(
                user_points[users], selected_user_gates, domain
            )
            return self._multi_interest_pair_scores(
                interests,
                mixture,
                item_points[items],
                selected_user_gates,
                selected_item_gates,
                domain,
            )
        return -self._squared_metric(
            user_points[users],
            item_points[items],
            selected_user_gates,
            selected_item_gates,
            domain,
        )

    def full_sort_target(self, users: Tensor) -> Tensor:
        encoded = self._encoded_for_inference()
        target_users, target_items = encoded[2], encoded[3]
        target_user_gates, target_item_gates = encoded[6], encoded[7]
        selected_user_gates = (
            None if target_user_gates is None else target_user_gates[users]
        )
        catalog_gates = (
            None
            if target_item_gates is None
            else target_item_gates[: self.target_num_items]
        )
        if self.enable_how2:
            interests, mixture, _ = self._multi_interest_bundle(
                target_users[users], selected_user_gates, "target"
            )
            scores = self._multi_interest_full_scores(
                interests,
                mixture,
                target_items[: self.target_num_items],
                selected_user_gates,
                catalog_gates,
                "target",
            )
        else:
            scores = -self._pairwise_squared_metric(
                target_users[users],
                target_items[: self.target_num_items],
                selected_user_gates,
                catalog_gates,
                "target",
            )
        return scores.reshape(-1)

    @torch.no_grad()
    def gate_diagnostics(self) -> dict[str, Tensor]:
        """Return geometry, gate, and multi-interest health for epoch logging."""

        if not self.uses_product_manifold:
            raise RuntimeError("gate diagnostics require geometry_mode=product")
        was_training = self.training
        self.eval()
        try:
            encoded = self._forward_with_gates()
        finally:
            self.train(was_training)
        result = {
            "gate_adaptive_scale": torch.tensor(
                self.gate_adaptive_scale,
                device=self.source_user_embedding.weight.device,
            ),
            "gate_prior_epochs": torch.tensor(
                self.gate_prior_epochs, device=self.source_user_embedding.weight.device
            ),
            "gate_ramp_epochs": torch.tensor(
                self.gate_ramp_epochs, device=self.source_user_embedding.weight.device
            ),
            "gate_initial_adaptive_scale": torch.tensor(
                self.gate_initial_adaptive_scale,
                device=self.source_user_embedding.weight.device,
            ),
            "component_prior": self.source_gate.component_prior.detach(),
            "source_spherical_curvature": self.source_spherical_curvature.detach(),
            "source_hyperbolic_curvature": self.source_curvature.detach(),
            "target_spherical_curvature": self.target_spherical_curvature.detach(),
            "target_hyperbolic_curvature": self.target_curvature.detach(),
            "how2_ramp_scale": torch.tensor(
                self.how2_loss_scale,
                device=self.source_user_embedding.weight.device,
            ),
            "how2_residual_scale": torch.tensor(
                self.how2_adaptive_scale,
                device=self.source_user_embedding.weight.device,
            ),
            "how3_loss_scale": torch.tensor(
                self.how3_loss_scale,
                device=self.source_user_embedding.weight.device,
            ),
        }
        gate_groups = {
            "source_user": encoded[4][self.source_active_users],
            "source_item": encoded[5][self.source_active_items],
            "target_user": encoded[6][self.target_active_users],
            "target_item": encoded[7][self.target_active_items],
        }
        for name, gates in gate_groups.items():
            result[f"{name}_gate"] = gates.mean(0)
            result[f"{name}_gate_argmax"] = torch.stack(
                [(gates.argmax(-1) == index).float().mean() for index in range(3)]
            )
            result[f"{name}_gate_entropy"] = -(
                gates.clamp_min(1e-12).log() * gates
            ).sum(-1).mean()
        if self.enable_how2 and self.shared_clusterer is not None:
            mixture_groups = {
                "source": self._cluster_assignments(
                    self.shared_clusterer,
                    encoded[0][self.source_active_users],
                    encoded[4][self.source_active_users],
                    "source",
                ),
                "target": self._cluster_assignments(
                    self.shared_clusterer,
                    encoded[2][self.target_active_users],
                    encoded[6][self.target_active_users],
                    "target",
                ),
            }
            usages = {}
            collapse_threshold = 0.90
            for domain, mixture in mixture_groups.items():
                safe_mixture = mixture.clamp_min(1e-12)
                usage = safe_mixture.mean(dim=0)
                usages[domain] = usage
                hard = mixture.argmax(dim=-1)
                hard_usage = torch.stack(
                    [
                        (hard == index).float().mean()
                        for index in range(mixture.size(-1))
                    ]
                )
                entropy = -(safe_mixture * safe_mixture.log()).sum(dim=-1)
                result[f"{domain}_interest_soft_usage"] = usage
                result[f"{domain}_interest_hard_usage"] = hard_usage
                result[f"{domain}_interest_mixture_entropy"] = entropy.mean()
                result[f"{domain}_interest_effective_count"] = entropy.exp().mean()
                result[f"{domain}_interest_max_probability"] = (
                    mixture.max(dim=-1).values.mean()
                )
                result[f"{domain}_interest_collapse_rate"] = (
                    mixture.max(dim=-1).values >= collapse_threshold
                ).float().mean()
                result[f"{domain}_active_interest_count"] = (
                    usage >= (0.5 / mixture.size(-1))
                ).float().sum()

            midpoint = 0.5 * (usages["source"] + usages["target"])
            result["interest_usage_js"] = 0.5 * (
                (
                    usages["source"]
                    * (
                        usages["source"].clamp_min(1e-12).log()
                        - midpoint.clamp_min(1e-12).log()
                    )
                ).sum()
                + (
                    usages["target"]
                    * (
                        usages["target"].clamp_min(1e-12).log()
                        - midpoint.clamp_min(1e-12).log()
                    )
                ).sum()
            )
            result["num_interests"] = torch.tensor(
                self.shared_clusterer.num_clusters,
                device=self.source_user_embedding.weight.device,
            )
            result["interest_collapse_probability_threshold"] = torch.tensor(
                collapse_threshold, device=self.source_user_embedding.weight.device
            )
            result["interest_center_separation_margin"] = torch.tensor(
                self.interest_separation_margin,
                device=self.source_user_embedding.weight.device,
            )

            centers_tangent = self.shared_clusterer.centers_tangent.detach()
            center_indices = torch.triu_indices(
                centers_tangent.size(0), centers_tangent.size(0), offset=1,
                device=centers_tangent.device,
            )
            center_gates = self.source_gate.component_prior.to(
                centers_tangent
            ).expand(centers_tangent.size(0), -1)
            for domain in ("source", "target"):
                centers = self._expmap_domain(centers_tangent, domain)
                pairwise = self._pairwise_squared_metric(
                    centers, centers, center_gates, center_gates, domain
                )
                distances = pairwise[
                    center_indices[0], center_indices[1]
                ].clamp_min(0.0).sqrt()
                result[f"{domain}_interest_center_distance_mean"] = distances.mean()
                result[f"{domain}_interest_center_distance_min"] = distances.min()
                result[f"{domain}_interest_center_distance_max"] = distances.max()
                result[f"{domain}_interest_center_below_margin_rate"] = (
                    distances < self.interest_separation_margin
                ).float().mean()
        return result

