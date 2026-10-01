import sys

import pytest
import torch
from rdkit import Chem

from loqi.featurize import mol_to_data, mols_to_data_list


def triples(data, low, high):
    return {
        (int(a), int(b), int(t))
        for (a, b), t in zip(data.edge_index.T, data.edge_attr, strict=True)
        if low <= int(t) <= high
    }


@pytest.mark.parametrize("app", [False, True])
@pytest.mark.parametrize("smiles", ["F[C@](Cl)(Br)I", "F[C@@](Cl)(Br)I"])
def test_app_and_api_default_to_fixed_lowest_cip_observer(app, smiles):
    from app.utils import mol_to_torch_geometric_simple

    convert = mol_to_torch_geometric_simple if app else mol_to_data
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    graph = convert(mol, smiles)
    assert triples(graph, 7, 7) == {(0, 2, 7), (2, 0, 7), (0, 3, 7), (3, 0, 7), (0, 4, 7), (4, 0, 7)}


@pytest.mark.parametrize("app", [False, True])
def test_carbon_only_keeps_carbon_and_ez_but_not_phosphorus(app):
    from app.utils import mol_to_torch_geometric_simple

    convert = mol_to_torch_geometric_simple if app else mol_to_data
    smi = "C[C@H](F)CC[P@](=O)(C)c1ccc(/C=C/F)cc1"
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    original = Chem.MolToSmiles(mol)
    default = convert(Chem.Mol(mol), smi)
    carbon = convert(Chem.Mol(mol), smi, c_chirality=True)
    all_edges = triples(default, 7, 8)
    c_edges = triples(carbon, 7, 8)
    assert len(all_edges) == 18 and len(c_edges) == 9 and c_edges < all_edges
    center = next(
        a for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    )
    neighbors = {a.GetIdx() for a in center.GetNeighbors()}
    assert all(a in neighbors and b in neighbors for a, b, _ in c_edges)
    assert triples(default, 1, 6) == triples(carbon, 1, 6)
    assert triples(carbon, 5, 6)
    assert torch.equal(default.x, carbon.x) and torch.equal(default.pos, carbon.pos)
    assert Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(carbon.mol))) == Chem.MolToSmiles(
        Chem.MolFromSmiles(original)
    )


@pytest.mark.parametrize("smi", ["C[Si@](F)(Cl)Br", "CC[P@](=O)(C)c1ccccc1", "C[S@](=O)(=N)CC"])
def test_noncarbon_only_and_replica_forwarding(smi):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    assert len(triples(mol_to_data(Chem.Mol(mol), smi), 7, 8)) == 9
    data = mols_to_data_list([mol], 2, c_chirality=True)
    assert len(data) == 2 and all(not triples(g, 7, 8) for g in data)


def test_api_forwards_carbon_only_flag(monkeypatch):
    from loqi import api

    original = api.featurize.mols_to_data_list
    called = []

    def capture(*args, **kwargs):
        called.append(kwargs["c_chirality"])
        return original(*args, **kwargs)

    monkeypatch.setattr(api.featurize, "mols_to_data_list", capture)

    def stop(*args, **kwargs):
        raise RuntimeError("stop before model loading")

    monkeypatch.setattr(api, "load_model", stop)
    with pytest.raises(RuntimeError, match="stop before model loading"):
        api.generate_conformers("CC[P@](=O)(C)c1ccccc1", 1, c_chirality=True)
    assert called == [True]


def test_sampling_cli_advertises_both_spellings(monkeypatch, capsys):
    from scripts import sample_conformers

    monkeypatch.setattr(sys, "argv", ["sample_conformers.py", "--help"])
    with pytest.raises(SystemExit) as exc:
        sample_conformers.main()
    assert exc.value.code == 0
    help = capsys.readouterr().out
    assert "--c-chirality" in help and "--c_chirality" in help


@pytest.mark.parametrize("flag", ["--c-chirality", "--c_chirality"])
def test_sampling_cli_forwards_flag(monkeypatch, flag):
    from types import SimpleNamespace

    from scripts import sample_conformers as cli

    monkeypatch.setattr(sys, "argv", ["sample_conformers.py", "--input", "CC", "--output", "unused.sdf", flag])
    monkeypatch.setattr(
        cli,
        "load_model",
        lambda *a, **kw: SimpleNamespace(
            model=None, config=SimpleNamespace(evaluation=SimpleNamespace()), default_batch_size=2
        ),
    )
    seen = []

    def capture(*args, **kwargs):
        seen.append(kwargs["c_chirality"])
        raise RuntimeError("stop before sampling")

    monkeypatch.setattr(cli, "mols_to_data_list", capture)
    with pytest.raises(RuntimeError, match="stop before sampling"):
        cli.main()
    assert seen == [True]


def test_app_batch_forwards_flag(monkeypatch):
    from app import utils

    seen = []

    def capture(*args, **kwargs):
        seen.append(kwargs["c_chirality"])
        raise RuntimeError("stop before sampling")

    monkeypatch.setattr(utils, "mol_to_torch_geometric_simple", capture)
    result = utils.generate_conformers_batch("CC", None, None, 1, c_chirality=True)
    assert seen == [True] and "stop before sampling" in str(result[-1])
