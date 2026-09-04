"""Drop-in replacements for the ``torch_scatter`` functions used in this package.

Implemented on top of :func:`torch_geometric.utils.scatter` /
:func:`torch_geometric.utils.softmax`, which fall back to pure ``torch``
(``index_add_`` / ``scatter_reduce``) when the compiled ``torch_scatter``
extension is not installed.  Signatures follow ``torch_scatter`` (``dim``
defaults to ``-1``; ``out=`` is not supported).
"""

from torch import Tensor
from torch_geometric.utils import scatter as _pyg_scatter
from torch_geometric.utils import softmax as _pyg_softmax


def _resolve_dim(src: Tensor, dim: int) -> int:
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
    return _pyg_scatter(src, index, dim=_resolve_dim(src, dim), dim_size=dim_size, reduce=reduce)


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
    return _pyg_softmax(src, index, num_nodes=dim_size, dim=_resolve_dim(src, dim))
