"""Tests for train-only molecule-size rebalancing."""

from collections import Counter

import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset

from megalodon.data.size_sampler import SizeTailSampler


def _dataset(*sizes):
    return [Data(x=torch.zeros((size, 1))) for size in sizes]


def test_size_tail_sampler_uses_exact_factors():
    sampler = SizeTailSampler(
        _dataset(5, 10, 11, 109, 110, 150),
        small_factor=5,
        large_factor=10,
        shuffle=False,
    )
    assert Counter(sampler) == Counter({0: 5, 1: 5, 2: 1, 3: 1, 4: 10, 5: 10})
    assert len(sampler) == 32


def test_rebalanced_continuation_uses_strict_size_boundaries():
    sampler = SizeTailSampler(_dataset(9, 10, 110, 111), small_max_nodes=9,
                              small_factor=100, large_min_nodes=111,
                              large_factor=10, shuffle=False)
    assert Counter(sampler) == Counter({0:100, 1:1, 2:1, 3:10})


def test_corrected_tails_include_diatomics_but_exclude_10_and_120():
    sampler=SizeTailSampler(_dataset(2,3,4,5,9,10,110,120,121,150),
                            small_max_nodes=9,small_factor=10,
                            large_min_nodes=121,large_factor=20,shuffle=False)
    assert Counter(sampler)==Counter({0:10,1:10,2:10,3:10,4:10,5:1,6:1,7:1,8:20,9:20})


def test_size_tail_sampler_shuffle_is_epoch_deterministic():
    dataset = _dataset(5, 50, 120)
    first = SizeTailSampler(dataset, small_factor=5, large_factor=5, seed=17)
    second = SizeTailSampler(dataset, small_factor=5, large_factor=5, seed=17)
    assert list(first) == list(second)
    first.set_epoch(1)
    assert list(first) != list(second)
    assert Counter(first) == Counter(second)


def test_tiny_tier_overrides_instead_of_multiplying_small_factor():
    sampler = SizeTailSampler(
        _dataset(2, 3, 4, 5, 9, 10, 110, 111),
        tiny_max_nodes=4, tiny_factor=100, small_max_nodes=9,
        small_factor=10, large_min_nodes=111, large_factor=20, shuffle=False,
    )
    assert Counter(sampler) == Counter({0:100, 1:100, 2:100, 3:10, 4:10, 5:1, 6:1, 7:20})


def test_tiny_tier_counts_explicit_hydrogens():
    from rdkit import Chem
    sizes = [Chem.AddHs(Chem.MolFromSmiles(smi)).GetNumAtoms()
             for smi in ('Cl', 'O', 'N', 'C', 'CO')]
    assert sizes == [2, 3, 4, 5, 6]
    sampler = SizeTailSampler(_dataset(*sizes), tiny_max_nodes=5,
                              tiny_factor=100, small_max_nodes=9,
                              small_factor=10, shuffle=False)
    assert Counter(sampler) == Counter({0:100, 1:100, 2:100, 3:100, 4:10})


@pytest.mark.parametrize('kwargs', [
    {'tiny_max_nodes':11}, {'tiny_max_nodes':-1}, {'tiny_max_nodes':2.5},
    {'tiny_factor':100}, {'tiny_max_nodes':4, 'tiny_factor':0},
])
def test_invalid_tiny_tier(kwargs):
    with pytest.raises(ValueError):
        SizeTailSampler(_dataset(3), **kwargs)


def test_size_tail_sampler_partitions_distributed_replicas():
    dataset = _dataset(5, 50, 120)
    rank_0 = SizeTailSampler(dataset, small_factor=2, large_factor=3, shuffle=False, num_replicas=2, rank=0)
    rank_1 = SizeTailSampler(dataset, small_factor=2, large_factor=3, shuffle=False, num_replicas=2, rank=1)
    combined = list(rank_0) + list(rank_1)
    assert Counter(combined) == Counter({0: 2, 1: 1, 2: 3})
    assert len(rank_0) == len(rank_1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"small_factor": 0},
        {"large_factor": 1.5},
        {"small_max_nodes": 110, "large_min_nodes": 110},
        {"num_replicas": 2, "rank": 2},
    ],
)
def test_size_tail_sampler_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SizeTailSampler(_dataset(20), **kwargs)


def test_in_memory_sampler_does_not_reconstruct_graphs(monkeypatch):
    dataset = InMemoryDataset()
    dataset._data, dataset.slices = InMemoryDataset.collate(_dataset(5, 50, 120))

    def forbidden_get(*args):
        raise AssertionError("Sampler must read sizes without materializing graphs")

    monkeypatch.setattr(dataset, "get", forbidden_get)
    sampler = SizeTailSampler(dataset, small_factor=5, large_factor=5, shuffle=False)
    assert Counter(sampler) == Counter({0: 5, 1: 1, 2: 5})
    assert dataset._data_list is None


def test_in_memory_subset_sizes_follow_subset_order(monkeypatch):
    dataset = InMemoryDataset()
    dataset._data, dataset.slices = InMemoryDataset.collate(_dataset(5, 50, 120))
    subset = dataset.index_select([2, 0, 2, 1])

    def forbidden_get(*args):
        raise AssertionError("Subset must not reconstruct graphs either")

    monkeypatch.setattr(subset, "get", forbidden_get)
    sampler = SizeTailSampler(subset, small_factor=2, large_factor=3, shuffle=False)
    assert Counter(sampler) == Counter({0: 3, 1: 2, 2: 3, 3: 1})


def test_distributed_sampler_is_not_wrapped_again():
    from lightning.pytorch import Trainer
    from torch_geometric.loader import DataLoader

    dataset = _dataset(5, 50, 120)
    sampler = SizeTailSampler(dataset, small_factor=2, large_factor=3, num_replicas=6, rank=1)
    loader = DataLoader(dataset, sampler=sampler)
    trainer = Trainer(accelerator="cpu", devices=6, strategy="ddp", logger=False,
                      enable_checkpointing=False, enable_progress_bar=False)
    assert not trainer._data_connector._requires_distributed_sampler(loader)
    assert trainer._data_connector._resolve_sampler(loader, shuffle=False) is sampler


def test_empty_in_memory_subset():
    dataset = InMemoryDataset()
    dataset._data, dataset.slices = InMemoryDataset.collate(_dataset(5, 50, 120))
    sampler = SizeTailSampler(dataset.index_select([]), num_replicas=6, rank=3)
    assert len(sampler) == 0
    assert list(sampler) == []


def test_diatomic_factor_is_exact_override_with_hydrogens():
    from rdkit import Chem
    sizes = [Chem.AddHs(Chem.MolFromSmiles(s)).GetNumAtoms()
             for s in ('[He]', 'Cl', 'F', 'Br', '[H][H]', 'O', 'N', 'C', 'CO')]
    sampler = SizeTailSampler(_dataset(*sizes, 10, 110, 111),
                              tiny_max_nodes=5, tiny_factor=100,
                              small_max_nodes=9, small_factor=10,
                              large_min_nodes=111, large_factor=20,
                              diatomic_factor=5000, shuffle=False)
    assert Counter(sampler) == Counter({0:100, 1:5000, 2:5000, 3:5000, 4:5000,
                                       5:100, 6:100, 7:100, 8:10, 9:1, 10:1, 11:20})


@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_invalid_diatomic_factor(value):
    with pytest.raises(ValueError, match='diatomic_factor'):
        SizeTailSampler(_dataset(2), diatomic_factor=value)


def test_diatomic_distributed_counts():
    dataset = _dataset(2, 3, 50, 111)
    samplers = [SizeTailSampler(dataset, diatomic_factor=5000,
                 tiny_max_nodes=5, tiny_factor=100, large_factor=20,
                 num_replicas=6, rank=rank, shuffle=False) for rank in range(6)]
    counts = Counter(i for sampler in samplers for i in sampler)
    assert counts == Counter({0:5003, 1:100, 2:1, 3:20})  # Three padding entries.
