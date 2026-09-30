"""Regression tests for LoQI stereochemical auxiliary edges."""

import pytest
from rdkit import Chem

from loqi.featurize import CHIRAL_EDGE_TYPES, mol_to_data, prepare_molecule
from megalodon.data.stereo import add_stereo_bonds
from megalodon.data import stereo


def _chiral_edges(data):
    return [
        (int(start), int(end), int(edge_type))
        for (start, end), edge_type in zip(data.edge_index.T, data.edge_attr, strict=True)
        if int(edge_type) in CHIRAL_EDGE_TYPES
    ]


def _stereo_data(smiles):
    mol, _ = prepare_molecule(smiles)
    return mol, mol_to_data(Chem.Mol(mol), smiles)


@pytest.mark.parametrize(
    ("smiles", "element", "expected_edges"),
    [
        ("C[C@H](O)c1ccccc1", "C", 9),
        ("C[Si@](F)(Cl)Br", "Si", 9),
        ("C[N@+](F)(Cl)Br", "N", 9),
        ("CC[P@](=O)(C)c1ccccc1", "P", 9),
        ("C[As@](Cl)c1ccccc1", "As", 0),
        ("C[S@@](=O)c1ccccc1", "S", 0),
        ("C[Se@@](=O)c1ccccc1", "Se", 0),
        ("C[S@](=O)(=N)CC", "S", 9),
    ],
)
def test_supported_tetrahedral_centers_have_clean_edges(smiles, element, expected_edges):
    mol, data = _stereo_data(smiles)
    centers = [
        mol.GetAtomWithIdx(index)
        for index, descriptor in Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
        if descriptor in {"R", "S"}
    ]
    assert [atom.GetSymbol() for atom in centers] == [element]

    edges = _chiral_edges(data)
    assert len(edges) == expected_edges
    assert len(edges) == len(set(edges))
    assert all(start != end for start, end, _ in edges)
    assert {edge_type for _, _, edge_type in edges} == (set(CHIRAL_EDGE_TYPES) if expected_edges else set())


def test_sulfoxide_enantiomers_are_both_unconditioned():
    _, r_data = _stereo_data("C[S@@](=O)c1ccccc1")
    _, s_data = _stereo_data("C[S@](=O)c1ccccc1")
    assert _chiral_edges(r_data) == _chiral_edges(s_data) == []


@pytest.mark.parametrize(
    "smiles",
    [
        "C[C@H](O)c1ccccc1",
        "CC[P@](=O)(C)c1ccccc1",
        "C[S@@](=O)c1ccccc1",
    ],
)
def test_stereo_edges_are_invariant_to_atom_renumbering(smiles):
    mol, _ = prepare_molecule(smiles)
    original = mol_to_data(Chem.Mol(mol), smiles)

    # RenumberAtoms takes a new-index -> old-index mapping.
    new_to_old = list(reversed(range(mol.GetNumAtoms())))
    renumbered = Chem.RenumberAtoms(mol, new_to_old)
    reordered = mol_to_data(renumbered, smiles)
    reordered_edges = {
        (new_to_old[start], new_to_old[end], edge_type) for start, end, edge_type in _chiral_edges(reordered)
    }
    assert reordered_edges == set(_chiral_edges(original))


@pytest.mark.parametrize("smiles", [
    "C[C@H](O)c1ccccc1", "C[C@@H](O)c1ccccc1",
    "CC[P@](=O)(C)c1ccccc1", "CC[P@@](=O)(C)c1ccccc1",
])
def test_opt_in_encoding_preserves_pretrained_four_ligand_convention(smiles):
    mol, _ = prepare_molecule(smiles)
    edges, types = add_stereo_bonds(mol, [7, 8], {}, from_3D=False, encoding="legacy_compatible")
    center = next(atom for atom in mol.GetAtoms() if atom.HasProp("_CIPCode"))
    neighbors = sorted(center.GetNeighbors(), key=lambda atom: int(atom.GetProp("_CIPRank")), reverse=True)
    a, b, c = [atom.GetIdx() for atom in neighbors[:3]]
    d = neighbors[-1].GetIdx()
    if center.GetProp("_CIPCode") == "S":
        a, b, c = c, b, a
    expected = {(a,d,7),(b,d,7),(c,d,7),(d,a,7),(d,b,7),(d,c,7),(b,a,8),(c,b,8),(a,c,8)}
    assert set(zip(edges[0].tolist(), edges[1].tolist(), types.tolist())) == expected


def test_opt_in_encoding_also_skips_sulfoxide_enantiomers():
    for smiles in ["C[S@](=O)c1ccccc1", "C[S@@](=O)c1ccccc1"]:
        mol, _ = prepare_molecule(smiles)
        edges, types = add_stereo_bonds(mol, [7, 8], {}, from_3D=False, encoding="legacy_compatible")
        assert edges is None and types is None


@pytest.mark.parametrize("encoding", ["canonical", "legacy_compatible"])
@pytest.mark.parametrize("smiles", [
    "Cl[P@TB17](Cl)(Cl)(Cl)Cl",
    "F[P@OH9-](F)(F)(F)(F)F",
    "O=C1O[I@SP1](O)c2ccccc21",
])
def test_non_tetrahedral_centers_are_not_encoded_even_with_cip_label(smiles, encoding, monkeypatch):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assign = stereo._assign_stereochemistry

    def assign_with_non_tetrahedral_cip(mol, from_3d):
        assign(mol, from_3d)
        for atom in mol.GetAtoms():
            if atom.GetChiralTag() in {
                Chem.ChiralType.CHI_TRIGONALBIPYRAMIDAL,
                Chem.ChiralType.CHI_OCTAHEDRAL,
                Chem.ChiralType.CHI_SQUAREPLANAR,
            }:
                atom.SetProp("_CIPCode", "R")

    monkeypatch.setattr(stereo, "_assign_stereochemistry", assign_with_non_tetrahedral_cip)
    edges, types = add_stereo_bonds(mol, [7, 8], {}, from_3D=False, encoding=encoding)
    assert edges is None and types is None


@pytest.mark.parametrize("encoding", ["canonical", "legacy_compatible"])
@pytest.mark.parametrize("unsupported", [
    "Cl[P@TB17](Cl)(Cl)(Cl)Cl",
    "F[P@OH9-](F)(F)(F)(F)F",
    "O=C1O[I@SP1](O)c2ccccc21",
])
def test_unsupported_centers_preserve_identity_and_supported_stereo(unsupported, encoding):
    mol = Chem.AddHs(Chem.MolFromSmiles(unsupported + ".C[S@](=O)c1ccccc1.C[C@H](O)F.F/C=C/F"))
    identity = Chem.MolToSmiles(mol)
    edges, types = add_stereo_bonds(
        mol, [7, 8], {Chem.BondStereo.STEREOE: 5, Chem.BondStereo.STEREOZ: 6},
        from_3D=False, encoding=encoding,
    )
    unsupported_atoms = set(Chem.GetMolFrags(mol)[0])
    assert not unsupported_atoms.intersection(edges.flatten().tolist())
    assert set(types.tolist()) == {5, 6, 7, 8}
    assert sum(int(t) in {7, 8} for t in types) == 9
    assert Chem.MolToSmiles(mol) == identity
