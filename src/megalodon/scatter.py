"""Scatter reductions using PyTorch Geometric.

These wrappers use PyTorch fallbacks when compiled extensions are unavailable.
The reduction axis defaults to the last dimension. Output buffers are unsupported.
"""

from torch import Tensor
from torch_geometric.utils import scatter as _pyg_scatter
from torch_geometric.utils import softmax as _pyg_softmax


def _normalize_axis(src: Tensor, dim: int) -> int:
    return src.dim() + dim if dim < 0 else dim


def scatter(
    src: Tensor,
    index: Tensor,
    dim: int = -1,
    out: Tensor | None = None,
    dim_size: int | None = None,
    reduce: str = "sum",
) -> Tensor:
    if out is not None:
        raise NotImplementedError("`out=` is not supported by the torch_geometric-backed scatter")
    return _pyg_scatter(src, index, dim=_normalize_axis(src, dim), dim_size=dim_size, reduce=reduce)


def scatter_sum(
    src: Tensor, index: Tensor, dim: int = -1, out: Tensor | None = None, dim_size: int | None = None
) -> Tensor:
    return scatter(src, index, dim=dim, out=out, dim_size=dim_size, reduce="sum")


scatter_add = scatter_sum


def scatter_mean(
    src: Tensor, index: Tensor, dim: int = -1, out: Tensor | None = None, dim_size: int | None = None
) -> Tensor:
    return scatter(src, index, dim=dim, out=out, dim_size=dim_size, reduce="mean")


def scatter_softmax(src: Tensor, index: Tensor, dim: int = -1, dim_size: int | None = None) -> Tensor:
    return _pyg_softmax(src, index, num_nodes=dim_size, dim=_normalize_axis(src, dim))
