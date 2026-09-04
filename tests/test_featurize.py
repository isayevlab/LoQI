"""SMILES validation, hydrogen handling and graph featurisation."""

import numpy as np
import pytest
import torch
from rdkit import Chem

from loqi import featurize
from loqi.featurize import (
    CHIRAL_EDGE_TYPES,
    EZ_EDGE_TYPES,
    build_sampling_loader,
    conformers_to_mol,
    legacy_stereo_perception,
    load_molecules,
    mol_to_data,
    mols_to_data_list,
    prepare_molecule,
)

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def test_prepare_molecule_adds_explicit_hydrogens():
    mol, canonical = prepare_molecule("CCO")
    assert canonical == "CCO"
    assert mol.GetNumAtoms() == 9
    assert sum(atom.GetNumImplicitHs() for atom in mol.GetAtoms()) == 0
    assert sum(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()) == 6


def test_prepare_molecule_can_skip_hydrogens():
    mol, _ = prepare_molecule("CCO", add_hs=False)
    assert mol.GetNumAtoms() == 3


def test_prepare_molecule_returns_canonical_smiles():
    _, canonical = prepare_molecule("OCC")
    assert canonical == "CCO"
    _, canonical = prepare_molecule("O=C(C)Oc1ccccc1C(O)=O")
    assert canonical == Chem.MolToSmiles(Chem.MolFromSmiles(ASPIRIN))


@pytest.mark.parametrize(
    ("smiles", "message"),
    [
        ("", "Empty"),
        ("C(C", "parse"),
        ("[Fe]", "Unsupported element"),
        ("C[CH2]", "Radical"),
    ],
)
def test_prepare_molecule_rejects_invalid_input(smiles, message):
    with pytest.raises(ValueError, match=message):
        prepare_molecule(smiles)


def test_prepare_molecule_warns_on_disconnected_fragments():
    with pytest.warns(UserWarning, match="Disconnected"):
        mol, _ = prepare_molecule("CCO.O")
    assert mol.GetNumAtoms() == 12


def test_mol_to_data_builds_graph_with_zero_coordinates():
    mol, _ = prepare_molecule("CCO")
    data = mol_to_data(Chem.Mol(mol), "CCO")
    assert data.x.dtype == torch.uint8
    assert data.x.shape == (9,)
    assert torch.equal(data.pos, torch.zeros(9, 3))
    assert data.edge_index.shape == (2, 16)  # 8 bonds, both directions
    assert data.edge_attr.dtype == torch.uint8
    assert torch.equal(data.edge_attr, torch.ones(16, dtype=torch.uint8))
    assert torch.equal(data.charges, torch.zeros(9, dtype=torch.int8))
    assert data.smiles == "CCO"


def test_mol_to_data_uses_3d_input_when_requested():
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(i), 0.0, 0.0))
    mol.AddConformer(conf)
    data = mol_to_data(Chem.Mol(mol), "CO", use_3d_input=True)
    assert torch.allclose(data.pos[:, 0], torch.arange(mol.GetNumAtoms(), dtype=torch.float))


def test_chiral_centre_adds_chiral_edges():
    with legacy_stereo_perception():
        mol, _ = prepare_molecule("C[C@H](N)O")
        data = mol_to_data(Chem.Mol(mol), "C[C@H](N)O")
    types = set(data.edge_attr.tolist())
    assert set(CHIRAL_EDGE_TYPES) <= types
    assert not (set(EZ_EDGE_TYPES.values()) & types)


@pytest.mark.parametrize(
    ("smiles", "edge_type"),
    [("C/C=C/C", EZ_EDGE_TYPES[Chem.BondStereo.STEREOE]), ("C/C=C\\C", EZ_EDGE_TYPES[Chem.BondStereo.STEREOZ])],
)
def test_double_bond_stereo_adds_ez_edges(smiles, edge_type):
    with legacy_stereo_perception():
        mol, _ = prepare_molecule(smiles)
        data = mol_to_data(Chem.Mol(mol), smiles)
    assert edge_type in set(data.edge_attr.tolist())


def test_stereo_edges_can_be_disabled():
    mol, _ = prepare_molecule("C[C@H](N)O")
    data = mol_to_data(Chem.Mol(mol), "C[C@H](N)O", use_stereo_bonds=False)
    assert set(data.edge_attr.tolist()) == {1}


def test_mols_to_data_list_replicates_and_tags_source_molecule():
    mols = [prepare_molecule("CCO")[0], prepare_molecule(ASPIRIN)[0]]
    data_list = mols_to_data_list(mols, 3)
    assert len(data_list) == 6
    assert [d.mol_idx for d in data_list] == [0, 0, 0, 1, 1, 1]
    assert all(d.mol is not mols[d.mol_idx] for d in data_list)  # featurisation works on copies
    assert [d.num_nodes for d in data_list] == [9, 9, 9, 21, 21, 21]


@pytest.mark.parametrize("atom_aware", [True, False])
def test_sampling_loader_visits_every_replica_once(atom_aware):
    mols = [prepare_molecule("CCO")[0], prepare_molecule(ASPIRIN)[0], prepare_molecule("c1ccccc1")[0]]
    data_list = mols_to_data_list(mols, 4)
    torch.manual_seed(0)
    loader = build_sampling_loader(data_list, 2, atom_aware_batching=atom_aware)
    seen = []
    for batch in loader:
        assert batch.mol_idx.shape == (batch.num_graphs,)
        seen.extend(batch.mol_idx.tolist())
    assert sorted(seen) == sorted(d.mol_idx for d in data_list)


def test_conformers_to_mol_sets_coordinates_and_ids():
    mol, _ = prepare_molecule("CCO")
    coords = [np.full((9, 3), float(k)) for k in range(2)]
    out = conformers_to_mol(mol, coords)
    assert out.GetNumConformers() == 2
    assert [c.GetId() for c in out.GetConformers()] == [0, 1]
    assert np.allclose(out.GetConformer(1).GetPositions(), 1.0)
    assert mol.GetNumConformers() == 0  # the template is not modified
    with pytest.raises(ValueError, match="shape"):
        conformers_to_mol(mol, [np.zeros((3, 3))])


def test_legacy_stereo_perception_is_restored():
    previous = Chem.GetUseLegacyStereoPerception()
    with legacy_stereo_perception(not previous):
        assert Chem.GetUseLegacyStereoPerception() is (not previous)
    assert Chem.GetUseLegacyStereoPerception() is previous


def test_load_molecules_from_smi_file_reports_bad_lines(tmp_path):
    path = tmp_path / "input.smi"
    path.write_text("CCO ethanol\nC(C\nc1ccccc1\n")
    mols, errors = load_molecules(str(path))
    assert [m.GetNumAtoms() for m in mols] == [9, 12]
    assert len(errors) == 1 and "line 2" in errors[0]


def test_load_molecules_from_smiles_string():
    mols, errors = load_molecules("CCO")
    assert len(mols) == 1 and errors == []
    with pytest.raises(ValueError):
        load_molecules("C(C")


def test_supported_elements_exported():
    assert {"H", "C", "N", "O", "F", "S", "Cl", "Br", "I"} <= featurize.SUPPORTED_ELEMENTS
