# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Spatial downscaler with a patchify encoder and an unpatchify decoder, as in a diffusion transformer (DiT).

Every grid point belongs to one hidden node (its patch) and holds a fixed slot in it. Patchify is the exact,
injective linear read of the patch: the hidden token is ``W @ patch`` and nothing else (no attention encoder,
no encoder edges). Unpatchify is one head per slot on the processed token, tied at initialisation to the
transpose of the read, so that before training each point's head returns its own noised target up to the
decoder LayerNorm scale and the processor's residual.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch import nn
from torch.utils.checkpoint import checkpoint

from anemoi.models.layers.graph_provider import NoOpGraphProvider
from anemoi.models.models.transport_encoder_processor_decoder import AnemoiTransportSpatialDownscalerModelEncProcDec
from anemoi.utils.config import DotDict


def _unit_vectors(coords_rad: np.ndarray) -> np.ndarray:
    lat, lon = coords_rad[:, 0], coords_rad[:, 1]
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)


def build_patch_assignment(
    points_rad: np.ndarray, nodes_rad: np.ndarray, max_points: int, num_candidates: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Assign each point to its nearest node (at most ``max_points`` per node) and a slot (azimuth rank)."""
    xyz_p, xyz_n = _unit_vectors(points_rad), _unit_vectors(nodes_rad)
    num_points, num_nodes = len(xyz_p), len(xyz_n)
    _, cand = cKDTree(xyz_n).query(xyz_p, k=min(num_candidates, num_nodes))
    cand = cand.reshape(num_points, -1)
    node = cand[:, 0].copy()
    counts = np.bincount(node, minlength=num_nodes)

    for n in np.flatnonzero(counts > max_points):
        pts = np.flatnonzero(node == n)
        dist = np.linalg.norm(xyz_p[pts] - xyz_n[n], axis=1)
        for p in pts[np.lexsort((pts, -dist))][: counts[n] - max_points]:
            for m in cand[p, 1:]:
                if counts[m] < max_points:
                    node[p], counts[n], counts[m] = m, counts[n] - 1, counts[m] + 1
                    break
            else:
                raise ValueError(f"Node {n} is full and point {p} has no neighbour with room; raise max_points.")

    lat_n, lon_n = nodes_rad[node, 0], nodes_rad[node, 1]
    lat_p, lon_p = points_rad[:, 0], points_rad[:, 1]
    dlon = lon_p - lon_n
    east = np.sin(dlon) * np.cos(lat_p)
    north = np.cos(lat_n) * np.sin(lat_p) - np.sin(lat_n) * np.cos(lat_p) * np.cos(dlon)
    azimuth = np.arctan2(east, north)
    dist = np.linalg.norm(xyz_p - xyz_n[node], axis=1)

    order = np.lexsort((np.arange(num_points), dist, azimuth, node))
    first = np.ones(num_points, dtype=bool)
    first[1:] = node[order][1:] != node[order][:-1]
    group_start = np.maximum.accumulate(np.where(first, np.arange(num_points), 0))
    slot = np.empty(num_points, dtype=np.int64)
    slot[order] = np.arange(num_points) - group_start
    return node.astype(np.int64), slot


def orthonormal_read_basis(width: int, num_slots: int, n_in: int, seed: int = 0) -> torch.Tensor:
    """Read matrix ``(width, num_slots * n_in)`` with orthonormal columns, so the read is injective."""
    if num_slots * n_in > width:
        raise ValueError(f"{num_slots} slots x {n_in} inputs exceed the token width {width}.")
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(width, num_slots * n_in, generator=generator, dtype=torch.float64))
    return q.float()


def _point_rows(node: torch.Tensor, num_nodes: int, bse: int) -> torch.Tensor:
    offsets = torch.arange(bse, device=node.device) * num_nodes
    return (node[None, :] + offsets[:, None]).reshape(-1)


class Patchify(nn.Module):
    """Patchify: ``token = W @ patch``, with ``patch`` the slot-padded points of a node.

    ``W`` has orthonormal columns, so ``W^T token`` returns every point's input columns exactly. It replaces the
    attention encoder; the hidden node's own attributes are not read.
    """

    def __init__(self, node: np.ndarray, slot: np.ndarray, num_nodes: int, read_weight: torch.Tensor):
        super().__init__()
        self.register_buffer("node", torch.from_numpy(node), persistent=False)
        self.register_buffer("slot", torch.from_numpy(slot), persistent=False)
        self.num_nodes = num_nodes
        self.num_slots = int(slot.max()) + 1
        self.hidden_dim = read_weight.shape[0]
        self.weight = nn.Parameter(read_weight.clone())
        self.n_in = read_weight.shape[1] // self.num_slots

    def forward(self, x: tuple[torch.Tensor, torch.Tensor], **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        group = kwargs.get("model_comm_group")
        if kwargs["shard_info"].src_nodes is not None or (group is not None and group.size() > 1):
            raise NotImplementedError("Patchify needs the whole grid on one GPU.")
        x_points = x[0]
        if x_points.shape[0] % self.node.numel():
            raise ValueError(f"{x_points.shape[0]} rows are not a multiple of the {self.node.numel()}-point grid.")
        bse = x_points.shape[0] // self.node.numel()
        patch = x_points.new_zeros(bse * self.num_nodes, self.num_slots, self.n_in)
        patch[_point_rows(self.node, self.num_nodes, bse), self.slot.repeat(bse)] = x_points
        token = F.linear(patch.flatten(1), self.weight.to(patch.dtype))
        return x_points, token


class Unpatchify(nn.Module):
    """Unpatchify: ``out_i = P_s A_s^T t + Q_s gelu(B_s^T t + b_s) + c_s``, ``t = LN(token of node(i))``, ``s = slot(i)``.

    ``A_s`` starts as the slot's read matrix, ``P_s`` selects the noised target and ``Q_s`` is zero.
    """

    def __init__(
        self,
        node: np.ndarray,
        slot: np.ndarray,
        num_nodes: int,
        read_weight: torch.Tensor,
        target_cols: torch.Tensor,
        hidden: int = 256,
    ):
        super().__init__()
        width = read_weight.shape[0]
        num_slots = int(slot.max()) + 1
        n_in, n_out = read_weight.shape[1] // num_slots, len(target_cols)
        self.register_buffer("node", torch.from_numpy(node), persistent=False)
        self.register_buffer("slot", torch.from_numpy(slot), persistent=False)
        self.num_nodes = num_nodes
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.A = nn.Parameter(read_weight.reshape(width, num_slots, n_in).permute(1, 0, 2).clone())
        P = torch.zeros(num_slots, n_in, n_out)
        P[:, target_cols, torch.arange(n_out)] = 1.0
        self.P = nn.Parameter(P)
        self.c = nn.Parameter(torch.zeros(num_slots, n_out))
        self.B = nn.Parameter(torch.randn(num_slots, width, hidden) / width**0.5)
        self.b = nn.Parameter(torch.zeros(num_slots, hidden))
        self.Q = nn.Parameter(torch.zeros(num_slots, hidden, n_out))

    def _heads(self, token: torch.Tensor) -> torch.Tensor:
        t = self.norm(token)
        lin = torch.einsum("nw,swo->nso", t, self.A @ self.P) + self.c
        h = F.gelu(torch.einsum("nw,swh->nsh", t, self.B) + self.b)
        return lin + torch.einsum("nsh,sho->nso", h, self.Q)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor], **kwargs) -> torch.Tensor:
        token = x[0]
        if self.training and torch.is_grad_enabled():
            out = checkpoint(self._heads, token, use_reentrant=False)
        else:
            out = self._heads(token)
        bse = token.shape[0] // self.num_nodes
        return out[_point_rows(self.node, self.num_nodes, bse), self.slot.repeat(bse)]


class AnemoiTransportPatchifyDownscalerModelEncProcDec(AnemoiTransportSpatialDownscalerModelEncProcDec):
    """Spatial downscaler whose encoder is Patchify and whose decoder is Unpatchify.

    The parent builds the configured encoder and decoder mappers and their edge providers first (the schema needs
    them); they are then replaced, so no attention mapper and no data-hidden edge is left in the model.
    """

    def _build_networks(self, model_config: DotDict) -> None:
        super()._build_networks(model_config)
        cfg = model_config.get("patchify") or {}
        if len(self.target_datasets) != 1:
            raise ValueError(f"The patchify downscaler takes one target, got {self.target_datasets}.")
        target = self.target_datasets[0]
        anchor = self._anchor_of_target(target)
        hidden = self._graph_name_hidden

        node, slot = build_patch_assignment(
            self._graph_data[anchor].x.double().cpu().numpy(),
            self._graph_data[hidden].x.double().cpu().numpy(),
            cfg.get("max_points", 14),
        )
        num_nodes = self.node_attributes.num_nodes[hidden]
        num_slots = int(slot.max()) + 1
        n_in = self.input_dim[anchor]
        read_weight = orthonormal_read_basis(
            self.latent_aggregator.hidden_dim, num_slots, n_in, cfg.get("read_seed", 0)
        )

        n_history = self.n_step_input * sum(self.num_input_channels[d] for d in self._encoder_input_datasets(anchor))
        target_cols = torch.arange(n_history, n_history + self.output_dim[target])

        encoder_name, decoder_name = self.dataset2encoder[anchor], self.dataset2decoder[target]
        self.encoder[encoder_name] = Patchify(node, slot, num_nodes, read_weight)
        self.decoder[decoder_name] = Unpatchify(
            node, slot, num_nodes, read_weight, target_cols, cfg.get("head_hidden", 256)
        )
        self.encoder_graph_provider[anchor] = NoOpGraphProvider()
        self.decoder_graph_provider[target] = NoOpGraphProvider()
