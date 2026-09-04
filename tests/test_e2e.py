"""End-to-end conformer generation with the released ``loqi`` checkpoint (CPU).

Runs only with ``pytest -m slow`` and only when a verified checkpoint is already in the cache
(``$LOQI_CACHE_DIR`` or ``~/.cache/loqi``; ``loqi download`` puts it there).
"""

import time

import numpy as np
import pytest
from rdkit import Chem

from loqi import generate_conformers, load_model
from loqi.cli import main
from loqi.registry import MODELS, default_cache_dir, sha256sum

pytestmark = pytest.mark.slow

SMILES = ["CCO", "CC(=O)Oc1ccccc1C(=O)O"]
N_CONFS = 3


@pytest.fixture(scope="module")
def loaded():
    path = default_cache_dir() / "loqi.ckpt"
    if not path.is_file() or sha256sum(path) != MODELS["loqi"].sha256:
        pytest.skip(f"no verified checkpoint at {path}; run `loqi download` first")
    t0 = time.perf_counter()
    model = load_model("loqi", device="cpu")
    print(f"\nload_model: {time.perf_counter() - t0:.1f} s")
    return model


def test_generate_conformers_cpu(loaded):
    t0 = time.perf_counter()
    mols = generate_conformers(SMILES, N_CONFS, model=loaded, seed=0)
    print(f"generate_conformers({len(SMILES)} molecules x {N_CONFS}): {time.perf_counter() - t0:.1f} s")

    assert len(mols) == len(SMILES)
    for smiles, mol in zip(SMILES, mols, strict=True):
        reference = Chem.MolFromSmiles(smiles)
        assert mol.GetNumAtoms() == Chem.AddHs(reference).GetNumAtoms()
        assert mol.GetNumConformers() + mol.GetIntProp("loqi_failed") == N_CONFS
        assert mol.GetNumConformers() >= 1
        assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == Chem.MolToSmiles(reference)
        for conf in mol.GetConformers():
            pos = conf.GetPositions()
            assert np.isfinite(pos).all()
            for bond in mol.GetBonds():
                length = np.linalg.norm(pos[bond.GetBeginAtomIdx()] - pos[bond.GetEndAtomIdx()])
                assert 0.7 < length < 2.5, f"{smiles}: bond length {length:.2f} A"


def test_same_seed_is_reproducible(loaded):
    first = generate_conformers("CCO", 2, model=loaded, seed=3)[0]
    second = generate_conformers("CCO", 2, model=loaded, seed=3)[0]
    for a, b in zip(first.GetConformers(), second.GetConformers(), strict=True):
        assert np.allclose(a.GetPositions(), b.GetPositions())


def test_cli_sample_writes_sdf(loaded, tmp_path):
    out = tmp_path / "confs.sdf"
    t0 = time.perf_counter()
    rc = main(["sample", "--smiles", "CCO", "--n-confs", "2", "--output", str(out), "--device", "cpu"])
    print(f"cli sample: {time.perf_counter() - t0:.1f} s")
    assert rc == 0
    records = [m for m in Chem.SDMolSupplier(str(out), removeHs=False)]
    assert len(records) == 2
    assert all(m.GetNumAtoms() == 9 for m in records)
    assert [m.GetIntProp("loqi_conformer_id") for m in records] == [0, 1]
    assert records[0].GetProp("_Name") == "CCO"
