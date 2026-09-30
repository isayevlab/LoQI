"""Ensure size-tail weighting is restricted to the training loader."""

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch_geometric.data import Data

from megalodon.data.molecule_datamodule import MoleculeDataModule
from megalodon.data.size_sampler import SizeTailSampler


def _module():
    module = MoleculeDataModule.__new__(MoleculeDataModule)
    module.data_loader_type = "midi"
    module.train_size_sampling = {
        "small_max_nodes": 10,
        "small_factor": 5,
        "large_min_nodes": 110,
        "large_factor": 5,
        "seed": 42,
    }
    module.sampler_kwargs = {}
    module.pin_memory = False
    return module


def test_only_training_loader_uses_size_tail_sampler():
    dataset = [Data(x=torch.zeros((size, 1))) for size in (5, 50, 120)]
    module = _module()

    train_loader = module._create_dataloader(dataset, batch_size=16, shuffle=True, is_train=True)
    val_loader = module._create_dataloader(dataset, batch_size=16, shuffle=True)
    test_loader = module._create_dataloader(dataset, batch_size=16, shuffle=False)

    assert isinstance(train_loader.sampler, SizeTailSampler)
    assert isinstance(val_loader.sampler, RandomSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)
    assert len(train_loader.sampler) == 11
    assert len(val_loader.sampler) == len(test_loader.sampler) == 3


def test_training_sampler_uses_distributed_rank(monkeypatch):
    dataset = [Data(x=torch.zeros((size, 1))) for size in (5, 50, 120)]
    module = _module()
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)

    loader = module._create_dataloader(
        dataset, batch_size=16, shuffle=True, is_train=True
    )

    assert isinstance(loader.sampler, SizeTailSampler)
    assert loader.sampler.num_replicas == 2
    assert loader.sampler.rank == 1
    assert len(loader.sampler) == 6


def test_disabled_tail_sampling_returns_to_normal_shuffle():
    dataset = [Data(x=torch.zeros((size, 1))) for size in (5, 50, 120)]
    module = _module()
    module.train_size_sampling = {}

    loader = module._create_dataloader(dataset, batch_size=16, shuffle=True, is_train=True)

    assert isinstance(loader.sampler, RandomSampler)
    assert not isinstance(loader.sampler, SizeTailSampler)
    assert len(loader.sampler) == len(dataset)
    assert sorted(loader.sampler) == [0, 1, 2]
