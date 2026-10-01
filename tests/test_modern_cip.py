"""Modern CIP ordering must agree across preprocessing, API and app paths."""

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from app.utils import mol_to_torch_geometric_simple
from data_processing.process_chembl3d import _convert_record
from data_processing.process_sdf import read_sdf_graphs
from data_processing.utils_data import save_pyg_dataset
from loqi.featurize import mol_to_data
from megalodon.data import stereo
from megalodon.data.stereo import add_stereo_bonds
from megalodon.metrics.preserved_stereo import get_stereochemistry_descriptor


def triples(graph):
    return {
        (int(a), int(b), int(t))
        for (a, b), t in zip(graph.edge_index.T, graph.edge_attr, strict=True)
        if int(t) in (7, 8)
    }


@pytest.mark.parametrize(
    "smiles,labels",
    [
        ("C[C@H]1CC[C@H](C)CC1", "rr"),
        ("C[C@H]1CC[C@@H](C)CC1", "ss"),
    ],
)
def test_ring_all_paths_and_geometry(smiles, labels):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=17) == 0
    coords = mol.GetConformer().GetPositions()
    api = mol_to_data(Chem.Mol(mol), smiles)
    app = mol_to_torch_geometric_simple(Chem.Mol(mol), smiles)
    dataset = _convert_record(Chem.Mol(mol), coords, "example")
    assert len(triples(api)) == 18
    assert triples(api) == triples(app) == triples(dataset)
    assert get_stereochemistry_descriptor(Chem.Mol(mol))[:2] == (labels, labels)
    edges, attr = add_stereo_bonds(mol, [7, 8], {}, from_3D=False)
    result = set(zip(edges[0].tolist(), edges[1].tolist(), attr.tolist(), strict=True))
    for atom in mol.GetAtoms():
        if not atom.HasProp("_CIPNeighborOrder"):
            continue
        order = list(atom.GetPropsAsDict(includePrivate=True, includeComputed=True)["_CIPNeighborOrder"])
        d = order[-1]
        assert mol.GetAtomWithIdx(d).GetSymbol() == "H"
        assert {(d, v, 7) for v in order[:3]} | {(v, d, 7) for v in order[:3]} <= result
        a = order[0]
        b = next(v for v in order[:3] if (a, v, 8) in result)
        c = next(v for v in order[:3] if v not in (a, b))
        assert np.linalg.det(coords[[a, b, c]] - coords[d]) > 0


def test_original_preprocessing_serializes_modern_edges_and_geometry(tmp_path):
    graphs = []
    for smi in ("CCO", "C[C@H](O)F", "C[C@H]1CC[C@H](C)CC1", "F/C=C/F"):
        mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        assert AllChem.EmbedMolecule(mol, randomSeed=17) == 0
        graph = _convert_record(mol, mol.GetConformer().GetPositions(), smi)
        reference = mol_to_data(Chem.Mol(mol), smi, use_3d_input=True)
        assert triples(graph) == triples(reference)
        assert torch.equal(graph.pos, reference.pos)
        graphs.append(graph)
    target = tmp_path / "train_h.pt"
    save_pyg_dataset(graphs, target)
    data, slices = torch.load(target, weights_only=False)
    assert len(data.mol) == len(graphs)
    for i, graph in enumerate(graphs):
        lo, hi = map(int, slices["edge_attr"][i : i + 2])
        nlo, nhi = map(int, slices["pos"][i : i + 2])
        assert torch.equal(graph.edge_index, data.edge_index[:, lo:hi])
        assert torch.equal(graph.edge_attr, data.edge_attr[lo:hi])
        assert torch.equal(graph.pos, data.pos[nlo:nhi])
        assert data.chemblid[i] == graph.chemblid
        assert Chem.MolToSmiles(data.mol[i]) == Chem.MolToSmiles(graph.mol)


@pytest.mark.parametrize("smiles", ["C[C@H]1CC[C@H](C)CC1", "C[C@H]1CC[C@@H](C)CC1"])
def test_original_sdf_preprocessing_encodes_pseudoasymmetric_centers(tmp_path, smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=17) == 0
    mol.SetProp("_Name", "ring")
    source = tmp_path / "ring.sdf"
    with Chem.SDWriter(str(source)) as writer:
        writer.write(mol)
    graphs, failed = read_sdf_graphs(source)
    assert failed == 0 and len(graphs) == 1
    assert graphs[0].chemblid == "ring"
    assert len(triples(graphs[0])) == 18
    assert triples(graphs[0]) == triples(mol_to_data(Chem.Mol(mol), smiles))


def test_budget_retry_keeps_all_dependent_ring_centers(monkeypatch):
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H]1CC[C@@H](C)CC1"))
    expected = add_stereo_bonds(Chem.Mol(mol), [7, 8], {}, from_3D=False)
    original = stereo.rdCIPLabeler.AssignCIPLabels

    def bounded(mol, **kwargs):
        assert len(kwargs["atomsToLabel"]) == 2
        if kwargs["maxRecursiveIterations"] < 1000000000:
            raise RuntimeError("Max Iterations Exceeded in CIP label calculation")
        return original(mol, **kwargs)

    monkeypatch.setattr(stereo.rdCIPLabeler, "AssignCIPLabels", bounded)
    actual = add_stereo_bonds(mol, [7, 8], {}, from_3D=False)
    assert torch.equal(expected[0], actual[0]) and torch.equal(expected[1], actual[1])


def test_cip_resource_failure_is_warned_and_marked(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Max Iterations Exceeded in CIP label calculation")

    monkeypatch.setattr(stereo.rdCIPLabeler, "AssignCIPLabels", fail)
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)F"))
    with pytest.warns(RuntimeWarning, match="unresolved atom indices"):
        assert add_stereo_bonds(mol, [7, 8], {}, from_3D=False) == (None, None)
    assert mol.GetAtomWithIdx(1).HasProp(stereo.CIP_FAILURE_PROP)


def test_unexpected_cip_errors_still_fail(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("unexpected library failure")

    monkeypatch.setattr(stereo.rdCIPLabeler, "AssignCIPLabels", fail)
    with pytest.raises(RuntimeError, match="unexpected"):
        add_stereo_bonds(Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)F")), [7, 8], {}, from_3D=False)


def test_completed_centers_and_ez_survive_another_center_failure(monkeypatch):
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)F.C[C@H](O)Cl.F/C=C/F"))
    centers = [a.GetIdx() for a in mol.GetAtoms() if stereo.is_supported_tetrahedral_center(a)]
    original = stereo.rdCIPLabeler.AssignCIPLabels

    def partial(mol, **kwargs):
        original(mol, atomsToLabel=[centers[0]], bondsToLabel=[], maxRecursiveIterations=1000000)
        raise RuntimeError("Digraph generation failed: more than 100000nodes found.")

    monkeypatch.setattr(stereo.rdCIPLabeler, "AssignCIPLabels", partial)
    with pytest.warns(RuntimeWarning, match="resource limit"):
        ix, attr = add_stereo_bonds(
            mol, [7, 8], {Chem.BondStereo.STEREOE: 5, Chem.BondStereo.STEREOZ: 6}, from_3D=False
        )
    assert int(((attr == 7) | (attr == 8)).sum()) == 9
    assert {5, 6} <= set(attr.tolist())
    assert not mol.GetAtomWithIdx(centers[0]).HasProp(stereo.CIP_FAILURE_PROP)
    assert mol.GetAtomWithIdx(centers[1]).HasProp(stereo.CIP_FAILURE_PROP)


def test_original_preprocessing_keeps_molecule_and_ez_on_cip_limit(monkeypatch):
    smi = "C[C@H](O)F.F/C=C/F"
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    assert AllChem.EmbedMolecule(mol, randomSeed=17) == 0
    coords = mol.GetConformer().GetPositions()
    expected = _convert_record(Chem.Mol(mol), coords, "example")

    def fail(*args, **kwargs):
        raise RuntimeError("Digraph generation failed: more than 100000nodes found.")

    monkeypatch.setattr(stereo.rdCIPLabeler, "AssignCIPLabels", fail)
    with pytest.warns(RuntimeWarning, match="resource limit"):
        graph = _convert_record(Chem.Mol(mol), coords, "example")
    assert graph.chemblid == "example" and graph.mol.GetNumAtoms() == mol.GetNumAtoms()
    assert not triples(graph)
    keep = expected.edge_attr < 7
    assert torch.equal(graph.edge_index, expected.edge_index[:, keep])
    assert torch.equal(graph.edge_attr, expected.edge_attr[keep])
    assert torch.equal(graph.pos, expected.pos)
    assert graph.mol.GetAtomWithIdx(1).HasProp(stereo.CIP_FAILURE_PROP)
