"""Numerics of megalodon.scatter (torch_geometric-backed torch_scatter replacements) against index_add_ references."""

import pytest
import torch

from megalodon import scatter as S

DIM_SIZE = 7
EMPTY_SEGMENTS = [2, 4, 6]


def ref_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype)
    return out.index_add_(0, index, src)


def ref_count(index: torch.Tensor, dim_size: int) -> torch.Tensor:
    return torch.zeros(dim_size).index_add_(0, index, torch.ones(index.numel()))


@pytest.fixture
def case():
    torch.manual_seed(0)
    src = torch.randn(12, 4, dtype=torch.float64)
    index = torch.tensor([0, 0, 1, 3, 3, 3, 5, 5, 5, 5, 0, 1])
    return src, index


def test_scatter_sum_matches_index_add(case):
    src, index = case
    out = S.scatter(src, index, dim=0, dim_size=DIM_SIZE, reduce="sum")
    assert out.shape == (DIM_SIZE, 4)
    assert torch.allclose(out, ref_sum(src, index, DIM_SIZE))
    assert torch.allclose(S.scatter_sum(src, index, dim=0, dim_size=DIM_SIZE), out)
    assert torch.allclose(S.scatter_add(src, index, dim=0, dim_size=DIM_SIZE), out)
    # torch_scatter accepted "add" as an alias of "sum"; nextmol/jodo call it that way.
    assert torch.allclose(S.scatter(src, index, 0, reduce="add", dim_size=DIM_SIZE), out)


def test_scatter_mean_matches_reference_and_zeroes_empty_segments(case):
    src, index = case
    out = S.scatter_mean(src, index, dim=0, dim_size=DIM_SIZE)
    expected = ref_sum(src, index, DIM_SIZE) / ref_count(index, DIM_SIZE).clamp(min=1).unsqueeze(-1)
    assert torch.allclose(out, expected)
    assert torch.equal(out[EMPTY_SEGMENTS], torch.zeros(len(EMPTY_SEGMENTS), 4, dtype=src.dtype))
    assert torch.isfinite(out).all()


def test_dim_size_is_inferred_from_index(case):
    src, index = case
    out = S.scatter_mean(src, index, dim=0)
    assert out.shape[0] == int(index.max()) + 1


def test_default_dim_is_last_axis_for_1d_input(case):
    _, index = case
    # denoising_models.py counts atoms per molecule with scatter_add(ones, batch) and the default dim=-1.
    counts = S.scatter_add(torch.ones_like(index), index)
    assert torch.equal(counts, torch.bincount(index))


def test_negative_dim_is_resolved_against_src(case):
    src, index = case
    src_t = src.T.contiguous()  # (4, 12): scatter along the last axis
    out = S.scatter(src_t, index, dim=-1, dim_size=DIM_SIZE, reduce="sum")
    assert out.shape == (4, DIM_SIZE)
    assert torch.allclose(out, ref_sum(src, index, DIM_SIZE).T)


def test_scatter_softmax_sums_to_one_per_segment(case):
    src, index = case
    out = S.scatter_softmax(src, index, dim=0, dim_size=DIM_SIZE)
    assert out.shape == src.shape
    assert (out > 0).all()
    sums = ref_sum(out, index, DIM_SIZE)
    occupied = ref_count(index, DIM_SIZE) > 0
    assert torch.allclose(sums[occupied], torch.ones_like(sums[occupied]))
    assert torch.equal(sums[~occupied], torch.zeros_like(sums[~occupied]))


def test_scatter_softmax_matches_per_segment_softmax(case):
    src, index = case
    out = S.scatter_softmax(src, index, dim=0, dim_size=DIM_SIZE)
    expected = torch.empty_like(src)
    for segment in index.unique():
        mask = index == segment
        expected[mask] = torch.softmax(src[mask], dim=0)
    assert torch.allclose(out, expected)


def test_scatter_softmax_1d_default_dim():
    logits = torch.tensor([1.0, 2.0, 3.0, 0.5, 0.5])
    index = torch.tensor([0, 0, 0, 1, 1])
    out = S.scatter_softmax(logits, index)
    assert torch.allclose(out[:3], torch.softmax(logits[:3], dim=0))
    assert torch.allclose(out[3:], torch.tensor([0.5, 0.5]))


def test_out_argument_is_rejected(case):
    src, index = case
    with pytest.raises(NotImplementedError):
        S.scatter(src, index, dim=0, out=torch.zeros(DIM_SIZE, 4, dtype=src.dtype))
