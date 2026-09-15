import torch
from megalodon.scatter import scatter_mean


def zero_com_pyg(coords, batch):
    """Zero center of mass per graph.

    Works for [N, 3], [N, 3, S], or any shape where dim-0 aligns with *batch*.
    """
    com = scatter_mean(coords, batch, dim=0)
    return coords - com[batch]


def find_reverse_edges(edge_index):
    """For each directed edge return the index of its reverse.

    Assumes every edge (i, j) has a matching (j, i) in *edge_index*.
    """
    src, dst = edge_index
    max_id = int(max(src.max(), dst.max())) + 1
    fwd = src.long() * max_id + dst.long()
    rev = dst.long() * max_id + src.long()
    sorted_fwd, sort_idx = fwd.sort()
    return sort_idx[torch.searchsorted(sorted_fwd, rev)]
