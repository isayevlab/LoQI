import math
from unittest.mock import Mock

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from megalodon.metrics import molecule_metrics_aimnet2 as aimnet
from megalodon.metrics.conformer_evaluation_callback import (
    ConformerEvaluationCallback,
    mean_available_metrics,
)


def molecule(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


@pytest.mark.parametrize("smiles,compute,idx", [
    ("[He]", aimnet.compute_bond_lengths_diff, 1),
    ("[H][H]", aimnet.compute_bond_angles_diff, 2),
    ("O", aimnet.compute_torsion_angles_diff, 3),
    ("C", aimnet.compute_torsion_angles_diff, 3),
])
def test_missing_geometry_is_unavailable(smiles, compute, idx):
    mol = molecule(smiles)
    assert aimnet.collect_geometry([(mol, mol)], compute) == {}
    assert math.isnan(aimnet.compute_distance([(mol, mol)], idx, compute))


def test_empty_geometry_does_not_dilute_populated_measurement():
    water, ethanol = molecule("O"), molecule("CCO")
    def compute(pair):
        return {} if pair[0].GetNumAtoms() == water.GetNumAtoms() else {(1, 1, 1, 1): ([4., 8.], 2)}
    assert aimnet.compute_distance([(water, water), (ethanol, ethanol)], 3, compute) == 6.
    assert aimnet.compute_distance([(ethanol, ethanol)], 3, aimnet.compute_torsion_angles_diff) == 0.


@pytest.mark.parametrize("chunked", [False, True])
def test_empty_preparation(chunked):
    assert aimnet.prepare_for_aimnet_chunked([], chunked=chunked) == ([], [])
    assert aimnet.prepare_for_aimnet([])["coord"].shape == (0, 0, 3)


@pytest.mark.parametrize("opt", [False, True])
@pytest.mark.parametrize("reference", [False, True])
@pytest.mark.parametrize("molecules", [[], [None], [Chem.Mol()], [Chem.MolFromSmiles("C.C")]])
def test_empty_or_all_invalid_never_invokes_aimnet(monkeypatch, opt, reference, molecules):
    metric = object.__new__(aimnet.MoleculeAIMNet2Metrics)
    metric.opt_metrics, metric.device = opt, "cpu"
    metric.model = Mock(side_effect=AssertionError("AIMNet must not be called"))
    metric.compute_optimized_metrics = Mock(side_effect=AssertionError("Optimization must not run"))
    monkeypatch.setattr(aimnet, "prepare_for_aimnet_chunked", Mock(side_effect=AssertionError("No batches")))
    refs = list(molecules) if reference else None
    result = metric(molecules, reference_molecules=refs, return_molecules=True)
    assert set(result[0]) == set(metric.empty_values(opt, reference))
    assert all(math.isnan(value) for value in result[0].values())
    assert result[1] == []
    if opt:
        assert result[2] == [] and result[3].numel() == 0
    metric.model.assert_not_called()
    metric.compute_optimized_metrics.assert_not_called()


def test_empty_energy_batch_never_invokes_model():
    metric = object.__new__(aimnet.MoleculeAIMNet2Metrics)
    metric.model = Mock(side_effect=AssertionError("AIMNet must not be called"))
    energy, forces = metric.calculate_energy_forces_batched(aimnet.prepare_for_aimnet([]))
    assert energy.shape == (0,) and forces.shape == (0, 0, 3)
    metric.model.assert_not_called()


def test_empty_optimizer_never_invokes_model(monkeypatch):
    metric = object.__new__(aimnet.MoleculeAIMNet2Metrics)
    optimize = Mock(side_effect=AssertionError("Optimization must not run"))
    monkeypatch.setattr(aimnet, "group_opt", optimize)
    metrics = {}
    result = metric.compute_optimized_metrics([], [], [], torch.empty(0), metrics, torch.empty(0))
    assert result[0] == result[1] == [] and result[2].numel() == 0
    assert set(metrics) == set(metric.empty_values(True, True))
    optimize.assert_not_called()


@pytest.mark.parametrize("molecules", [[], [None], [Chem.MolFromSmiles("C.C")]])
def test_callback_does_not_load_aimnet_for_empty_or_invalid(monkeypatch, molecules):
    constructor = Mock(side_effect=AssertionError("AIMNet must not be loaded"))
    monkeypatch.setattr(aimnet.MoleculeAIMNet2Metrics, "__init__", constructor)
    callback = ConformerEvaluationCallback(
        compute_3D_metrics=False, compute_stereo_metrics=False,
        energy_metrics_args={"opt_metrics": True},
    )
    result = callback.evaluate_molecules(molecules, list(molecules), "cpu")
    assert set(result) == set(aimnet.MoleculeAIMNet2Metrics.empty_values(True, True))
    constructor.assert_not_called()


def test_distributed_average_excludes_missing_observations(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    def all_reduce(totals):
        # Other rank supplies a=4, b=6, c=missing.
        totals += torch.tensor([[4., 6., 0.], [1., 1., 0.]], dtype=totals.dtype)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    result = mean_available_metrics({"a": float("nan"), "b": 2., "c": float("nan")}, "cpu")
    assert result["a"] == result["b"] == 4.
    assert math.isnan(result["c"])


def test_callback_logs_keys_in_identical_collective_order():
    callback = ConformerEvaluationCallback(
        compute_3D_metrics=False, compute_stereo_metrics=False,
        energy_metrics_args={"opt_metrics": True},
    )
    module = Mock(device="cpu")
    callback.on_validation_epoch_end(None, module)
    metrics = module.log_dict.call_args.args[0]
    assert list(metrics) == sorted(metrics)
    assert module.log_dict.call_args.kwargs == {"sync_dist": True}
