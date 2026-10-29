# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from anemoi.models.models.patchify_downscaler import AnemoiTransportPatchifyDownscalerModelEncProcDec
from anemoi.models.models.patchify_downscaler import Patchify
from anemoi.models.models.patchify_downscaler import Unpatchify
from anemoi.models.models.patchify_downscaler import build_patch_assignment
from anemoi.models.models.patchify_downscaler import orthonormal_read_basis
from tests.models.test_transport_spatial_downscaler import _make_downscaler_config
from tests.models.test_transport_spatial_downscaler import _make_downscaler_graph
from tests.models.test_transport_spatial_downscaler import _make_downscaler_indices
from tests.models.test_transport_spatial_downscaler import _run_real_predict_step

GRID, HIDDEN, MAX_POINTS = 40, 8, 10


def _coords(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(0.6, 0.9, n), rng.uniform(0.0, 0.3, n)], axis=1)


def _read_and_head(n_in: int = 5, width: int = 64, target_cols=(2, 3)):
    node, slot = build_patch_assignment(_coords(GRID, 0), _coords(HIDDEN, 1), MAX_POINTS)
    weight = orthonormal_read_basis(width, int(slot.max()) + 1, n_in)
    read = Patchify(node, slot, HIDDEN, weight)
    head = Unpatchify(node, slot, HIDDEN, weight, torch.tensor(target_cols))
    return read, head, n_in


def test_read_is_injective() -> None:
    read, _, n_in = _read_and_head()
    pairs = set(zip(read.node.tolist(), read.slot.tolist()))
    assert len(pairs) == GRID and np.bincount(read.node.numpy()).max() <= MAX_POINTS
    assert torch.allclose(read.weight.T @ read.weight, torch.eye(read.weight.shape[1]), atol=1e-5)

    bse = 3  # batch x ensemble: rows are (sample, point), tokens are (sample, node)
    x = torch.randn(bse * GRID, n_in)
    x_out, token = read((x, None), shard_info=SimpleNamespace(src_nodes=None))
    assert x_out is x and token.shape == (bse * HIDDEN, read.hidden_dim)
    patch = (token @ read.weight).reshape(bse, HIDDEN, read.num_slots, n_in)
    expected = x.reshape(bse, GRID, n_in)
    for sample in range(bse):
        assert torch.allclose(patch[sample][read.node, read.slot], expected[sample], atol=1e-5)


def test_step0_head_is_tied_to_the_read() -> None:
    """Without the decoder LayerNorm the head inverts the read on the noised columns exactly."""
    read, head, n_in = _read_and_head()
    head.norm = nn.Identity()
    x = torch.randn(GRID, n_in)
    _, token = read((x, None), shard_info=SimpleNamespace(src_nodes=None))
    out = head((token, None))
    assert torch.allclose(out, x[:, [2, 3]], atol=1e-5)


def test_toy_forward() -> None:
    config = _make_downscaler_config(num_channels=128)
    config.model._target_ = "anemoi.models.models.AnemoiTransportPatchifyDownscalerModelEncProcDec"
    config.patchify = {"max_points": MAX_POINTS, "head_hidden": 16}
    graph = _make_downscaler_graph(grid=GRID, hidden=HIDDEN)
    for name in ("out_hres", "in_lres", "in_hres"):
        graph[name].x = torch.from_numpy(_coords(GRID, 0)).float()
    graph["hidden"].x = torch.from_numpy(_coords(HIDDEN, 1)).float()

    model = AnemoiTransportPatchifyDownscalerModelEncProcDec(
        model_config=config,
        data_indices=_make_downscaler_indices(),
        statistics={name: None for name in ("out_hres", "in_lres", "in_hres")},
        n_step_input=1,
        n_step_output=1,
        graph_data=graph,
    )
    assert isinstance(model.encoder["enc0"], Patchify)
    assert isinstance(model.decoder["dec0"], Unpatchify)
    assert model.encoder_graph_provider["in_lres"].edge_dim == 0
    assert model.decoder_graph_provider["out_hres"].edge_dim == 0

    batch = {"in_lres": torch.randn(2, 1, GRID, 2), "in_hres": torch.randn(2, 1, GRID, 1)}
    out = _run_real_predict_step(model, batch)
    assert out["out_hres"].shape == (2, 1, 1, GRID, 2)
    assert torch.isfinite(out["out_hres"]).all()


def test_toy_forward_leaves_no_parameter_without_gradient() -> None:
    """No attention mapper, edge or hidden-node parameter is left dead behind the patchify ends."""
    config = _make_downscaler_config(num_channels=128)
    config.model._target_ = "anemoi.models.models.AnemoiTransportPatchifyDownscalerModelEncProcDec"
    config.patchify = {"max_points": MAX_POINTS, "head_hidden": 16}
    graph = _make_downscaler_graph(grid=GRID, hidden=HIDDEN)
    for name in ("out_hres", "in_lres", "in_hres"):
        graph[name].x = torch.from_numpy(_coords(GRID, 0)).float()
    graph["hidden"].x = torch.from_numpy(_coords(HIDDEN, 1)).float()
    model = AnemoiTransportPatchifyDownscalerModelEncProcDec(
        model_config=config,
        data_indices=_make_downscaler_indices(),
        statistics={name: None for name in ("out_hres", "in_lres", "in_hres")},
        n_step_input=1,
        n_step_output=1,
        graph_data=graph,
    )
    model.train()
    x = {"in_lres": torch.randn(1, 1, 1, GRID, 2), "in_hres": torch.randn(1, 1, 1, GRID, 1)}
    noised = {"out_hres": torch.randn(1, 1, 1, GRID, 2)}
    sigma = {name: torch.full((1, 1, 1, 1, 1), 0.5) for name in ("in_lres", "out_hres")}
    out = model.forward(x, noised, sigma)
    out["out_hres"].square().mean().backward()
    # The toy processor has no conditional LayerNorm, so only the noise-conditioning MLP may stay unused here.
    dead = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None and not name.startswith("noise_cond_mlp.")
    ]
    assert not dead, dead
