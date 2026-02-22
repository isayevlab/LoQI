"""
Batch-size sweep: find optimal throughput and GPU utilization.

Samples GPU SM% and memory-BW% every second via `nvidia-smi dmon` while
generation runs, then reports min/mean/max for each batch size.

Usage:
  conda run -n loqi python scripts/batch_size_sweep.py \
      --config scripts/conf/loqi/loqi.yaml \
      --ckpt /home/olexandr/geoopt/data/loqi.ckpt \
      --dataset_root /home/olexandr/geoopt/data/LoQI/chembl3d_stereo \
      --smiles_pickle /home/olexandr/geoopt/data/LoQI/chembl3d_stereo/processed/test_smiles.pickle
"""

import sys, time, argparse, pickle, random, subprocess, threading, re
sys.path.insert(0, "src")

import torch
import numpy as np
from omegaconf import OmegaConf

from megalodon.inference import generate_conformers, validate_smiles


# ── GPU sampling via nvidia-smi dmon ─────────────────────────────────────────

class GpuMonitor:
    """Background thread that reads nvidia-smi dmon output every ~1s."""

    def __init__(self, gpu_idx: int = 0):
        self.gpu_idx = gpu_idx
        self._proc = None
        self._thread = None
        self._samples: list[dict] = []
        self._running = False

    def start(self):
        self._samples = []
        self._running = True
        # -s um: sm + memory utilization; -d 1: 1-second interval; -i N: GPU index
        cmd = ["nvidia-smi", "dmon", "-s", "um", "-d", "1", "-i", str(self.gpu_idx)]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        for line in self._proc.stdout:
            if not self._running:
                break
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            # format: gpu_idx  sm  mem  enc  dec  jpg  ofa  fb  bar1  ccpm
            if len(parts) >= 3 and parts[0].isdigit():
                try:
                    self._samples.append({"sm": int(parts[1]), "mem": int(parts[2]),
                                          "fb_mb": int(parts[7]) if len(parts) > 7 else 0})
                except ValueError:
                    pass

    def stop(self) -> dict:
        self._running = False
        if self._proc:
            self._proc.terminate()
            self._proc.wait()
        if self._thread:
            self._thread.join(timeout=2)

        if not self._samples:
            return {"sm_mean": 0, "sm_max": 0, "mem_mean": 0, "mem_max": 0, "fb_max_mb": 0, "n": 0}

        sm  = [s["sm"]    for s in self._samples]
        mem = [s["mem"]   for s in self._samples]
        fb  = [s["fb_mb"] for s in self._samples]
        return {
            "sm_mean":  int(np.mean(sm)),
            "sm_max":   int(np.max(sm)),
            "mem_mean": int(np.mean(mem)),
            "mem_max":  int(np.max(mem)),
            "fb_max_mb": int(np.max(fb)),
            "n": len(sm),
        }


# ── helpers ──────────────────────────────────────────────────────────────────

def sync():
    torch.cuda.synchronize()

def gpu_device():
    return torch.cuda.current_device()


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


def load_smiles(smiles_pickle, n_mols, seed=42):
    with open(smiles_pickle, "rb") as f:
        all_smiles = pickle.load(f)
    rng = random.Random(seed)
    rng.shuffle(all_smiles)

    validated = []
    atom_counts = []
    for smi in all_smiles:
        if len(validated) >= n_mols:
            break
        mol, err = validate_smiles(smi)
        if err is None:
            validated.append(smi)
            atom_counts.append(mol.GetNumAtoms())

    return validated, np.array(atom_counts)


def bar(val, max_val=100, width=20, fill="█", empty="░"):
    n = int(val / max_val * width)
    return fill * n + empty * (width - n)


def section(title):
    print(f"\n{'═'*66}")
    print(f"  {title}")
    print(f"{'═'*66}")


# ── sweep ────────────────────────────────────────────────────────────────────

def run_sweep(model, cfg, smiles_list, n_confs, gpu_idx, batch_sizes):
    results = []

    for bs in batch_sizes:
        # Skip if previous run OOM'd
        if results and results[-1].get("oom"):
            results.append({"bs": bs, "oom": True})
            continue

        mon = GpuMonitor(gpu_idx)
        torch.cuda.reset_peak_memory_stats()
        sync()

        try:
            mon.start()
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
        except torch.cuda.OutOfMemoryError:
            mon.stop()
            results.append({"bs": bs, "oom": True})
            torch.cuda.empty_cache()
            continue

        gpu_stats = mon.stop()
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        n_ok = result.n_success
        rate = n_ok / elapsed

        results.append({
            "bs": bs,
            "elapsed": elapsed,
            "rate": rate,
            "ms_per_conf": elapsed / n_ok * 1000,
            "n_ok": n_ok,
            "peak_mb": peak_mb,
            "sm_mean": gpu_stats["sm_mean"],
            "sm_max": gpu_stats["sm_max"],
            "mem_mean": gpu_stats["mem_mean"],
            "mem_max": gpu_stats["mem_max"],
            "n_samples": gpu_stats["n"],
            "oom": False,
        })

    return results


def print_results(results, n_mols, n_confs):
    best_rate = max((r["rate"] for r in results if not r.get("oom")), default=1)

    print(f"\n  {'bs':>5}  {'conf/s':>7}  {'ms/conf':>8}  "
          f"{'peak MB':>8}  {'SM% avg':>8}  {'SM% max':>8}  "
          f"{'BW% avg':>8}  {'throughput bar'}")
    print(f"  {'─'*5}  {'─'*7}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*20}")

    for r in results:
        bs = r["bs"]
        if r.get("oom"):
            print(f"  {bs:>5}  {'OOM':>7}")
            continue
        rel = r["rate"] / best_rate
        tbar = bar(rel, max_val=1.0, width=22)
        print(f"  {bs:>5}  {r['rate']:>7.1f}  {r['ms_per_conf']:>8.1f}  "
              f"{r['peak_mb']:>8.0f}  {r['sm_mean']:>8d}  {r['sm_max']:>8d}  "
              f"{r['mem_mean']:>8d}  {tbar} {rel*100:.0f}%")

    # Find optimal: best throughput per memory dollar (rate / peak_mb)
    valid = [r for r in results if not r.get("oom")]
    if not valid:
        return
    best_throughput = max(valid, key=lambda r: r["rate"])
    best_efficiency = max(valid, key=lambda r: r["rate"] / r["peak_mb"])

    print(f"\n  Optimal throughput:   batch_size={best_throughput['bs']}  "
          f"→ {best_throughput['rate']:.1f} conf/s  "
          f"(SM avg {best_throughput['sm_mean']}%  "
          f"peak {best_throughput['peak_mb']:.0f} MB)")
    print(f"  Optimal efficiency:   batch_size={best_efficiency['bs']}  "
          f"→ {best_efficiency['rate']:.1f} conf/s / {best_efficiency['peak_mb']:.0f} MB  "
          f"= {best_efficiency['rate']/best_efficiency['peak_mb']*1000:.2f} conf·s⁻¹·GB⁻¹")


def print_utilization_timeline(results):
    """Show how SM utilization changes with batch size."""
    section("GPU SM% utilization by batch size")
    valid = [r for r in results if not r.get("oom")]
    if not valid:
        return
    max_sm = max(r["sm_max"] for r in valid) or 1
    for r in valid:
        b = bar(r["sm_mean"], max_val=100, width=30)
        b2 = bar(r["sm_max"],  max_val=100, width=30)
        print(f"  bs={r['bs']:>4}  avg {r['sm_mean']:>3}% {b}  "
              f"max {r['sm_max']:>3}% {b2}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         required=True)
    parser.add_argument("--ckpt",           required=True)
    parser.add_argument("--dataset_root",   default=None)
    parser.add_argument("--smiles_pickle",  required=True)
    parser.add_argument("--n_mols",         type=int,   default=200,
                        help="Number of molecules to use per sweep run")
    parser.add_argument("--n_confs",        type=int,   default=1)
    parser.add_argument("--gpu",            type=int,   default=0,
                        help="GPU index for nvidia-smi monitoring")
    parser.add_argument("--batch_sizes",    type=str,   default="1,2,4,8,12,16,24,32,48,64,96,128",
                        help="Comma-separated batch sizes to sweep")
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    # ── Load molecules ────────────────────────────────────────────────────────
    section(f"Loading {args.n_mols} molecules")
    smiles_list, atom_counts = load_smiles(args.smiles_pickle, args.n_mols)
    print(f"  Loaded {len(smiles_list)} molecules")
    print(f"  Atom counts: min={atom_counts.min()}  max={atom_counts.max()}  "
          f"mean={atom_counts.mean():.1f}  median={np.median(atom_counts):.1f}")

    # Buckets
    for label, lo, hi in [("tiny ≤20", 0, 20), ("small 21-40", 21, 40),
                           ("medium 41-60", 41, 60), ("large >60", 61, 9999)]:
        n = ((atom_counts >= lo) & (atom_counts <= hi)).sum()
        print(f"  {label:14s}: {n:4d}  ({100*n/len(atom_counts):.1f}%)")

    # ── Load model ────────────────────────────────────────────────────────────
    section("Loading model")
    model, cfg = load_model(args.ckpt, args.config, args.dataset_root)
    gpu_name = torch.cuda.get_device_name(args.gpu)
    gpu_total_mb = torch.cuda.get_device_properties(args.gpu).total_memory / 1e6
    print(f"  GPU {args.gpu}: {gpu_name}  ({gpu_total_mb:.0f} MB total)")
    print(f"  Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # ── Warm-up ───────────────────────────────────────────────────────────────
    section("Warm-up (16 molecules, discarded)")
    _ = generate_conformers(smiles_list[:16], model, cfg, n_confs=1, batch_size=8)
    sync()
    print("  Done.")

    # ── Sweep: n_confs=1 ─────────────────────────────────────────────────────
    section(f"Sweep — {len(smiles_list)} mols × {args.n_confs} conf  "
            f"(batch sizes: {batch_sizes})")
    results = run_sweep(model, cfg, smiles_list, args.n_confs, args.gpu, batch_sizes)
    print_results(results, len(smiles_list), args.n_confs)
    print_utilization_timeline(results)

    # ── Repeat with n_confs=5 for multi-conf view ─────────────────────────────
    if args.n_confs == 1:
        section(f"Sweep — {len(smiles_list)} mols × 5 conf  (multi-conformer view)")
        results5 = run_sweep(model, cfg, smiles_list, 5, args.gpu, batch_sizes)
        print_results(results5, len(smiles_list), 5)

    section("Done")


if __name__ == "__main__":
    main()
