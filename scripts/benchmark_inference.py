"""
Benchmark script for the new megalodon.inference API.

Tests:
  1. Validation pipeline (bad SMILES, unsupported elements, etc.)
  2. Single-conformer batched generation (batch_size control)
  3. Multi-conformer generation (n_confs > 1 per molecule)
  4. FFD atom-count bin-packing vs fixed batch_size
  5. Timing breakdown at each stage
  6. Accuracy probe: SMILES round-trip identity, atom-count conservation

Usage:
  conda run -n loqi python scripts/benchmark_inference.py \
      --config scripts/conf/loqi/loqi.yaml \
      --ckpt /home/olexandr/geoopt/data/loqi.ckpt
"""

import sys, time, argparse
sys.path.insert(0, "src")

import torch
from omegaconf import OmegaConf
from rdkit import Chem

# ── Diverse drug-like SMILES ────────────────────────────────────────────────
SMILES_SET = {
    # Name → SMILES (all single-fragment, drug-like)
    "aspirin":      "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine":     "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "ibuprofen":    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol":  "CC(=O)Nc1ccc(O)cc1",
    "naproxen":     "COc1ccc2cc(ccc2c1)C(C)C(=O)O",
    "ciprofloxacin":"O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
    "metformin":    "CN(C)C(=N)NC(=N)N",
    "atorvastatin": "CC(C)c1c(C(=O)Nc2ccccc2F)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O",
    "lisinopril":   "NCCCC[C@@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O",
    "tamoxifen":    "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "morphine":     "OC1=CC=C2C[C@H]3N(CCc4cc5c(cc4O3)OCC5)C[C@@H]2C1",  # complex ring system
    "imatinib":     "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",  # big drug
    "benzene":      "c1ccccc1",  # tiny
    "dopamine":     "NCCc1ccc(O)c(O)c1",
    "serotonin":    "NCCc1c[nH]c2ccc(O)cc12",
    # Edge cases for validation
    "has_fluorine":      "Fc1ccccc1",
    "has_chlorine":      "Clc1ccccc1",
    "has_bromine":       "Brc1ccccc1",
    "has_phosphorus":    "OP(O)(O)=O",
    "has_sulfur":        "c1ccc(S)cc1",
}

BAD_SMILES = {
    "empty":        "",
    "garbage":      "NOT_A_SMILES_XYZ",
    "salt":         "CC(=O)O.[Na+]",
    "radical":      "[CH3]",            # radical (unlikely but valid test)
    "unsupported":  "[Pt](Cl)(Cl)(Cl)Cl",  # Pt not in vocab
}

ALL_GOOD_SMILES = list(SMILES_SET.values())
ALL_NAMES       = list(SMILES_SET.keys())


def section(title: str):
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")


def run_validation_tests():
    section("1. Validation Pipeline")
    from megalodon.inference import validate_smiles

    # Good SMILES
    good_pass = 0
    for name, smi in SMILES_SET.items():
        mol, err = validate_smiles(smi)
        if mol is not None and err is None:
            good_pass += 1
        else:
            print(f"  UNEXPECTED FAIL  [{name}]: {err}")
    print(f"  Good SMILES: {good_pass}/{len(SMILES_SET)} passed validation")

    # Bad SMILES
    bad_caught = 0
    for name, smi in BAD_SMILES.items():
        mol, err = validate_smiles(smi)
        if err is not None and mol is None:
            bad_caught += 1
            print(f"  [CAUGHT] {name}: {err[:80]}")
        else:
            print(f"  MISSED  {name}: expected failure but got mol")
    print(f"  Bad SMILES: {bad_caught}/{len(BAD_SMILES)} correctly rejected")


def load_model(ckpt_path: str, config_path: str, dataset_root: str = None):
    section("2. Model Loading")
    from megalodon.models.module import Graph3DInterpolantModel
    from megalodon.data.batch_preprocessor import BatchPreProcessor

    cfg = OmegaConf.load(config_path)
    if dataset_root is not None:
        OmegaConf.update(cfg, "data.dataset_root", dataset_root, merge=True)
    t0 = time.perf_counter()
    model = Graph3DInterpolantModel.load_from_checkpoint(
        ckpt_path,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=BatchPreProcessor(cfg.data.aug_rotations, cfg.data.scale_coords),
        strict=False,  # 'freqs' buffers added in inference-opt; not in older checkpoints
    )
    model = model.to("cuda").eval()
    t1 = time.perf_counter()
    print(f"  Model loaded in {t1-t0:.2f}s")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params/1e6:.1f}M")
    return model, cfg


def run_single_conf_benchmark(model, cfg, smiles_list, names):
    section("3. Single-Conformer Batched Generation")
    from megalodon.inference import generate_conformers

    for batch_size in [4, 8, 16]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = generate_conformers(
            smiles_list=smiles_list,
            model=model,
            cfg=cfg,
            n_confs=1,
            batch_size=batch_size,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        n_ok = result.n_success
        n_err = result.n_errors
        rate = n_ok / elapsed
        print(f"  batch_size={batch_size:2d} | {elapsed:6.2f}s | "
              f"{n_ok}/{len(smiles_list)} OK | {n_err} errors | "
              f"{rate:.1f} conformers/s")
        if result.errors:
            for e in result.errors:
                print(f"    SKIP [{names[e.index]}]: {e.error[:70]}")
    return result  # return last result for accuracy checks


def run_multi_conf_benchmark(model, cfg, smiles_list, names):
    section("4. Multi-Conformer Generation (n_confs=5)")
    from megalodon.inference import generate_conformers

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = generate_conformers(
        smiles_list=smiles_list,
        model=model,
        cfg=cfg,
        n_confs=5,
        batch_size=16,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_confs = result.n_success
    n_mol = len(result.conformers)
    print(f"  Generated {total_confs} conformers for {n_mol} molecules in {elapsed:.2f}s")
    print(f"  {elapsed/total_confs*1000:.1f} ms/conformer  |  {total_confs/elapsed:.1f} conformers/s")
    print(f"  Conformers per molecule:")
    for smi, mols in result.conformers.items():
        name = names[smiles_list.index(smi)] if smi in smiles_list else smi[:20]
        print(f"    {name}: {len(mols)} conformers")
    return result


def run_variable_nconfs_benchmark(model, cfg, smiles_list, names):
    section("5. Variable n_confs per molecule")
    from megalodon.inference import generate_conformers

    # Give small molecules more conformers, big ones fewer
    from megalodon.inference import validate_smiles
    n_confs_list = []
    for smi in smiles_list:
        mol, _ = validate_smiles(smi)
        n_atoms = mol.GetNumAtoms() if mol else 30
        n_confs_list.append(max(1, 10 - n_atoms // 10))

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = generate_conformers(
        smiles_list=smiles_list,
        model=model,
        cfg=cfg,
        n_confs=n_confs_list,
        batch_size=16,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total = result.n_success
    print(f"  {total} conformers total in {elapsed:.2f}s ({total/elapsed:.1f}/s)")
    print(f"  Per-mol requested vs generated:")
    for (smi, n_req), name in zip(zip(smiles_list, n_confs_list), names):
        n_gen = len(result.conformers.get(smi, []))
        print(f"    {name:20s}: requested={n_req}, generated={n_gen}")
    return result


def run_ffd_benchmark(model, cfg, smiles_list, names):
    section("6. FFD Bin-Packing vs Fixed batch_size")
    from megalodon.inference import generate_conformers, validate_smiles, ffd_pack_indices

    # Compute atom counts for reference
    print("  Atom counts per molecule:")
    total_atoms = 0
    for name, smi in zip(names, smiles_list):
        mol, _ = validate_smiles(smi)
        n = mol.GetNumAtoms() if mol else 0
        total_atoms += n
        print(f"    {name:20s}: {n:3d} atoms")
    print(f"  Total atoms: {total_atoms}, avg: {total_atoms/len(smiles_list):.1f}")

    # FFD with max_atoms_per_batch
    for max_atoms in [128, 256, 512]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = generate_conformers(
            smiles_list=smiles_list,
            model=model,
            cfg=cfg,
            n_confs=1,
            max_atoms_per_batch=max_atoms,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"  FFD max_atoms={max_atoms:4d} | {elapsed:6.2f}s | "
              f"{result.n_success}/{len(smiles_list)} OK")

    # Fixed batch_size for comparison
    for bs in [8, 16]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = generate_conformers(
            smiles_list=smiles_list,
            model=model,
            cfg=cfg,
            n_confs=1,
            batch_size=bs,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        print(f"  Fixed batch_size={bs:4d}   | {elapsed:6.2f}s | "
              f"{result.n_success}/{len(smiles_list)} OK")


def run_accuracy_checks(result, smiles_list, names):
    section("7. Accuracy Checks")
    from megalodon.inference import validate_smiles

    n_checked = 0
    n_correct_atoms = 0
    n_has_conf = 0
    n_3d_coords_nonzero = 0
    atom_errors = []

    for smi, conf_mols in result.conformers.items():
        ref_mol, _ = validate_smiles(smi)
        ref_atoms = ref_mol.GetNumAtoms() if ref_mol else None
        for i, gen_mol in enumerate(conf_mols):
            n_checked += 1
            if gen_mol is None:
                continue
            if gen_mol.GetNumConformers() > 0:
                n_has_conf += 1
                pos = gen_mol.GetConformer().GetPositions()
                if pos.std() > 0.1:
                    n_3d_coords_nonzero += 1
            gen_atoms = gen_mol.GetNumAtoms()
            if ref_atoms is not None and gen_atoms == ref_atoms:
                n_correct_atoms += 1
            elif ref_atoms is not None:
                name = names[smiles_list.index(smi)] if smi in smiles_list else smi[:20]
                atom_errors.append(f"    {name}: ref={ref_atoms}, gen={gen_atoms}")

    print(f"  Conformers checked:        {n_checked}")
    print(f"  Has 3D conformer:          {n_has_conf}/{n_checked}")
    print(f"  3D coords non-trivial:     {n_3d_coords_nonzero}/{n_has_conf}")
    print(f"  Correct atom count:        {n_correct_atoms}/{n_checked}")
    if atom_errors:
        print("  Atom count mismatches:")
        for e in atom_errors:
            print(e)

    # SMILES round-trip
    n_roundtrip = 0
    for smi, conf_mols in result.conformers.items():
        for gen_mol in conf_mols:
            if gen_mol is None:
                continue
            try:
                gen_smi = Chem.MolToSmiles(Chem.RemoveHs(gen_mol))
                ref_smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
                if gen_smi == ref_smi:
                    n_roundtrip += 1
            except Exception:
                pass
    print(f"  SMILES round-trip match:   {n_roundtrip}/{n_checked}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_root", default=None,
                        help="Override data.dataset_root in config (needed if config has stale path)")
    parser.add_argument("--quick", action="store_true",
                        help="Use only 6 small molecules for fast iteration")
    args = parser.parse_args()

    smiles_list = ALL_GOOD_SMILES
    names       = ALL_NAMES

    if args.quick:
        keep = ["aspirin", "caffeine", "ibuprofen", "paracetamol", "naproxen", "dopamine"]
        idx = [ALL_NAMES.index(k) for k in keep if k in ALL_NAMES]
        smiles_list = [ALL_GOOD_SMILES[i] for i in idx]
        names       = [ALL_NAMES[i] for i in idx]
        print(f"[quick mode] Using {len(smiles_list)} small molecules")

    run_validation_tests()
    model, cfg = load_model(args.ckpt, args.config, dataset_root=args.dataset_root)

    result_single = run_single_conf_benchmark(model, cfg, smiles_list, names)
    result_multi  = run_multi_conf_benchmark(model, cfg, smiles_list, names)
    run_variable_nconfs_benchmark(model, cfg, smiles_list, names)
    run_ffd_benchmark(model, cfg, smiles_list, names)
    run_accuracy_checks(result_multi, smiles_list, names)

    section("Done")


if __name__ == "__main__":
    main()
