"""
Sustained performance test using real ChEMBL3D test-set SMILES.

Samples molecules across the full atom-count range, runs large batches,
measures GPU memory, throughput, and timing breakdowns.

Usage:
  conda run -n loqi python scripts/sustained_perf_test.py \
      --config scripts/conf/loqi/loqi.yaml \
      --ckpt /home/olexandr/geoopt/data/loqi.ckpt \
      --dataset_root /home/olexandr/geoopt/data/LoQI/chembl3d_stereo \
      --smiles_pickle /home/olexandr/geoopt/data/LoQI/chembl3d_stereo/processed/test_smiles.pickle \
      --n_mols 500 \
      --n_confs 1
"""

import sys, time, argparse, pickle, random, textwrap
sys.path.insert(0, "src")

import torch
import numpy as np
from omegaconf import OmegaConf
from rdkit import Chem

from megalodon.inference import generate_conformers, validate_smiles, ffd_pack_indices


# ── helpers ─────────────────────────────────────────────────────────────────

def gpu_mem_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    return 0.0

def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_model(ckpt_path, config_path, dataset_root):
    from megalodon.models.module import Graph3DInterpolantModel
    from megalodon.data.batch_preprocessor import BatchPreProcessor
    cfg = OmegaConf.load(config_path)
    if dataset_root:
        OmegaConf.update(cfg, "data.dataset_root", dataset_root, merge=True)
    model = Graph3DInterpolantModel.load_from_checkpoint(
        ckpt_path,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=BatchPreProcessor(cfg.data.aug_rotations, cfg.data.scale_coords),
        strict=False,
    )
    return model.to("cuda").eval(), cfg


def sample_smiles(smiles_pickle, n_mols, seed=42):
    """
    Load SMILES and stratify by atom count so we get a representative mix
    of tiny (≤20), small (21-40), medium (41-60), and large (>60 atom) mols.
    """
    with open(smiles_pickle, "rb") as f:
        all_smiles = pickle.load(f)

    rng = random.Random(seed)
    rng.shuffle(all_smiles)

    # Validate and count atoms
    validated = []
    for smi in all_smiles:
        if len(validated) >= n_mols * 4:   # over-sample then stratify
            break
        mol, err = validate_smiles(smi)
        if err is None:
            n_atoms = mol.GetNumAtoms()
            validated.append((smi, n_atoms))

    # Stratify: 25% per bracket
    buckets = {
        "tiny":   [s for s, n in validated if n <= 20],
        "small":  [s for s, n in validated if 21 <= n <= 40],
        "medium": [s for s, n in validated if 41 <= n <= 60],
        "large":  [s for s, n in validated if n > 60],
    }
    per_bucket = n_mols // 4
    result = []
    for label, pool in buckets.items():
        chosen = pool[:per_bucket]
        result.extend(chosen)
        print(f"  {label:6s}: {len(chosen):4d} molecules "
              f"(pool {len(pool)})")

    # Top up from any bucket if needed
    remainder = n_mols - len(result)
    if remainder > 0:
        all_valid = [s for s, _ in validated if s not in set(result)]
        result.extend(all_valid[:remainder])

    rng.shuffle(result)
    print(f"  Total sampled: {len(result)}")
    return result


def atom_count_stats(smiles_list):
    counts = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = Chem.AddHs(mol)
            counts.append(mol.GetNumAtoms())
    arr = np.array(counts)
    return arr


def print_histogram(counts, bins=8, label="Atom count distribution"):
    hist, edges = np.histogram(counts, bins=bins)
    peak = max(hist)
    bar_width = 30
    print(f"\n  {label}:")
    for i, (lo, hi, h) in enumerate(zip(edges, edges[1:], hist)):
        bar = "█" * int(h / peak * bar_width)
        print(f"  {lo:4.0f}–{hi:4.0f}: {bar:<{bar_width}} {h}")
    print(f"  min={counts.min():.0f}  max={counts.max():.0f}  "
          f"mean={counts.mean():.1f}  median={np.median(counts):.1f}")


def section(title):
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")


# ── benchmark runs ────────────────────────────────────────────────────────────

def run_batch_size_sweep(model, cfg, smiles_list, n_confs, label=""):
    section(f"Batch-size sweep — {len(smiles_list)} mols × {n_confs} conf  {label}")
    results = {}
    for bs in [8, 16, 32, 64]:
        reset_peak()
        sync()
        t0 = time.perf_counter()
        result = generate_conformers(
            smiles_list=smiles_list,
            model=model,
            cfg=cfg,
            n_confs=n_confs,
            batch_size=bs,
        )
        sync()
        elapsed = time.perf_counter() - t0
        peak_gb = gpu_mem_gb()
        n_ok = result.n_success
        rate = n_ok / elapsed
        results[bs] = dict(elapsed=elapsed, rate=rate, peak_gb=peak_gb,
                           n_ok=n_ok, n_err=result.n_errors)
        print(f"  batch_size={bs:3d} | {elapsed:7.2f}s | "
              f"{n_ok}/{len(smiles_list)*n_confs} OK | "
              f"{rate:6.1f} conf/s | peak GPU {peak_gb:.2f} GB")
    return results


def run_ffd_sweep(model, cfg, smiles_list, n_confs):
    section(f"FFD atom-count sweep — {len(smiles_list)} mols × {n_confs} conf")

    # Show what FFD bins look like at each cap
    mol_atoms = atom_count_stats(smiles_list)
    for cap in [256, 512, 1024, 2048]:
        bins = ffd_pack_indices(mol_atoms.tolist(), cap)
        sizes = [len(b) for b in bins]
        print(f"  cap={cap:5d}: {len(bins):3d} bins, "
              f"mols/bin min={min(sizes)} max={max(sizes)} avg={np.mean(sizes):.1f}")
    print()

    for cap in [256, 512, 1024, 2048]:
        reset_peak()
        sync()
        t0 = time.perf_counter()
        result = generate_conformers(
            smiles_list=smiles_list,
            model=model,
            cfg=cfg,
            n_confs=n_confs,
            max_atoms_per_batch=cap,
        )
        sync()
        elapsed = time.perf_counter() - t0
        peak_gb = gpu_mem_gb()
        n_ok = result.n_success
        print(f"  max_atoms={cap:5d} | {elapsed:7.2f}s | "
              f"{n_ok}/{len(smiles_list)*n_confs} OK | "
              f"{n_ok/elapsed:6.1f} conf/s | peak GPU {peak_gb:.2f} GB")


def run_multi_conf_scaling(model, cfg, smiles_list):
    section(f"Multi-conformer scaling — {len(smiles_list)} unique molecules")
    for n_confs in [1, 5, 10, 20]:
        reset_peak()
        sync()
        t0 = time.perf_counter()
        result = generate_conformers(
            smiles_list=smiles_list,
            model=model,
            cfg=cfg,
            n_confs=n_confs,
            batch_size=32,
        )
        sync()
        elapsed = time.perf_counter() - t0
        peak_gb = gpu_mem_gb()
        total_confs = result.n_success
        print(f"  n_confs={n_confs:3d} | {elapsed:7.2f}s | "
              f"{total_confs} conformers | "
              f"{total_confs/elapsed:6.1f} conf/s | "
              f"{elapsed/total_confs*1000:.1f} ms/conf | "
              f"peak GPU {peak_gb:.2f} GB")


def run_sustained_load(model, cfg, smiles_list, n_confs, batch_size, n_rounds=5):
    section(f"Sustained load — {n_rounds} rounds × {len(smiles_list)} mols × {n_confs} conf  "
            f"(batch_size={batch_size})")
    round_times = []
    total_confs = 0
    t_wall_start = time.perf_counter()
    for i in range(n_rounds):
        reset_peak()
        sync()
        t0 = time.perf_counter()
        result = generate_conformers(
            smiles_list=smiles_list,
            model=model,
            cfg=cfg,
            n_confs=n_confs,
            batch_size=batch_size,
        )
        sync()
        elapsed = time.perf_counter() - t0
        peak_gb = gpu_mem_gb()
        total_confs += result.n_success
        round_times.append(elapsed)
        print(f"  round {i+1}/{n_rounds}: {elapsed:.2f}s  "
              f"{result.n_success/elapsed:.1f} conf/s  "
              f"peak {peak_gb:.2f} GB")

    wall = time.perf_counter() - t_wall_start
    arr = np.array(round_times)
    print(f"\n  Rounds: mean={arr.mean():.2f}s  std={arr.std():.2f}s  "
          f"cv={arr.std()/arr.mean()*100:.1f}%")
    print(f"  Total: {total_confs} conformers in {wall:.1f}s  "
          f"({total_confs/wall:.1f} sustained conf/s)")


def run_accuracy_probe(model, cfg, smiles_list, n_confs=3):
    section(f"Accuracy probe — {len(smiles_list)} mols × {n_confs} conf")
    result = generate_conformers(
        smiles_list=smiles_list,
        model=model,
        cfg=cfg,
        n_confs=n_confs,
        batch_size=32,
    )

    n_checked = n_correct_atoms = n_has_3d = n_roundtrip = 0
    atom_errors = []

    for smi, conf_mols in result.conformers.items():
        ref_mol = Chem.AddHs(Chem.MolFromSmiles(smi))
        ref_n = ref_mol.GetNumAtoms()
        for gen_mol in conf_mols:
            n_checked += 1
            if gen_mol is None:
                continue
            if gen_mol.GetNumConformers() > 0:
                n_has_3d += 1
            if gen_mol.GetNumAtoms() == ref_n:
                n_correct_atoms += 1
            else:
                atom_errors.append((smi, ref_n, gen_mol.GetNumAtoms()))
            try:
                gen_smi = Chem.MolToSmiles(Chem.RemoveHs(gen_mol))
                if gen_smi == Chem.MolToSmiles(Chem.MolFromSmiles(smi)):
                    n_roundtrip += 1
            except Exception:
                pass

    print(f"  Conformers checked:       {n_checked}")
    print(f"  Has 3D conformer:         {n_has_3d}/{n_checked}  "
          f"({100*n_has_3d/max(n_checked,1):.1f}%)")
    print(f"  Correct atom count:       {n_correct_atoms}/{n_checked}  "
          f"({100*n_correct_atoms/max(n_checked,1):.1f}%)")
    print(f"  SMILES round-trip:        {n_roundtrip}/{n_checked}  "
          f"({100*n_roundtrip/max(n_checked,1):.1f}%)")
    if atom_errors:
        print(f"  Atom count mismatches ({len(atom_errors)}):")
        for smi, r, g in atom_errors[:5]:
            print(f"    {smi[:50]}: ref={r} gen={g}")
    print(f"  Errors (failed validation): {result.n_errors}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--smiles_pickle", required=True)
    parser.add_argument("--n_mols", type=int, default=500)
    parser.add_argument("--n_confs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    section(f"Setup — sampling {args.n_mols} molecules from ChEMBL3D test set")
    smiles_list = sample_smiles(args.smiles_pickle, args.n_mols, seed=args.seed)

    counts = atom_count_stats(smiles_list)
    print_histogram(counts, bins=10)

    print(f"\n  Loading model...")
    model, cfg = load_model(args.ckpt, args.config, args.dataset_root)
    n_params = sum(p.numel() for p in model.parameters())
    device = next(model.parameters()).device
    print(f"  {n_params/1e6:.1f}M params on {device} "
          f"({torch.cuda.get_device_name(0)})")

    # Warm-up: one small batch so CUDA kernels are compiled
    section("Warm-up (10 molecules)")
    _ = generate_conformers(smiles_list[:10], model, cfg, n_confs=1, batch_size=8)
    print("  Done.")

    run_batch_size_sweep(model, cfg, smiles_list, args.n_confs)
    run_ffd_sweep(model, cfg, smiles_list, args.n_confs)
    run_multi_conf_scaling(model, cfg, smiles_list[:100])  # 100 unique mols
    run_sustained_load(model, cfg, smiles_list, args.n_confs, batch_size=32, n_rounds=5)
    run_accuracy_probe(model, cfg, smiles_list[:200], n_confs=3)

    section("Done")


if __name__ == "__main__":
    main()
