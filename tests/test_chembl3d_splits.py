"""Tests for stable, leakage-free ChEMBL3D splits."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

MODULE_PATH = Path(__file__).parents[1] / "data_processing" / "process_chembl3d.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("process_chembl3d", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _record(mol_id, row):
    return MODULE.SelectedConformer(group=10, row=row, mol_id=mol_id, energy=float(row), stereo_id=0)


def test_parent_variants_never_cross_splits():
    records = [_record(f"CHEMBL{index}_0", index) for index in range(100)]
    records.extend([_record("CHEMBL7_1", 100), _record("CHEMBL7_2", 101)])
    splits = MODULE.split_conformers(records, seed=42)

    locations = {record.mol_id: split_name for split_name, split_records in splits.items() for record in split_records}
    assert locations["CHEMBL7_0"] == locations["CHEMBL7_1"] == locations["CHEMBL7_2"]
    assert MODULE.audit_split_parents(splits) == {
        name: len({MODULE.parent_mol_id(record.mol_id) for record in split}) for name, split in splits.items()
    }


def test_existing_hash_assignments_do_not_move_when_records_are_added():
    original = [_record(f"CHEMBL{index}_0", index) for index in range(100)]
    expanded = original + [_record(f"CHEMBL{index}_0", index) for index in range(100, 150)]

    def assignments(records):
        return {
            record.mol_id: split_name
            for split_name, split_records in MODULE.split_conformers(records, seed=7).items()
            for record in split_records
        }

    before = assignments(original)
    after = assignments(expanded)
    assert all(after[mol_id] == split for mol_id, split in before.items())


@pytest.mark.parametrize("smiles", ["F[C@](Cl)(Br)I", "F[C@@](Cl)(Br)I"])
def test_preprocessing_defaults_to_original_cip_edges(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=17) == 0
    graph = MODULE._convert_record(mol, mol.GetConformer().GetPositions(), "stereo-regression")
    actual = {
        (int(a), int(b), int(t))
        for (a, b), t in zip(graph.edge_index.T, graph.edge_attr, strict=True)
        if int(t) in {7, 8}
    }
    expected = {(0, 2, 7), (2, 0, 7), (0, 3, 7), (3, 0, 7), (0, 4, 7), (4, 0, 7)}
    cycle = {(3, 4, 8), (2, 3, 8), (4, 2, 8)}
    if graph.mol.GetAtomWithIdx(1).GetProp("_CIPCode") == "S":
        cycle = {(b, a, t) for a, b, t in cycle}
    assert actual == expected | cycle
