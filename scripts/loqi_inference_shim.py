"""LoQI inference shim — diffusion-sampler entry point.

Usage:
    python loqi_inference_shim.py \\
        --input INPUT.json --output_dir OUT/ --n_confs N \\
        --ckpt CKPT.pt --node_dist NODE.pickle [--seed 0]

CLI contract (stable; downstream consumers depend on defaults):
    --seed     RNG seed for diffusion sampling. Default 0. Pinned at default
               since 2026-05-12 for byte-reproducibility across re-runs.
               Any change to the default is a breaking API change for
               aimnet2-solv and torchcosmors_benchmarks.

Reads: INPUT.json (list[{inchikey, smiles}])
Writes: <inchikey>.conformers.sdf in OUT/, plus
        loqi_inference_summary.json for downstream consumers.

Note: seed must be set before model loading AND before the DataLoader
iterates. AdaptiveBatchSampler consumes one global-RNG draw per batch
(see megalodon/data/adaptive_dataloader.py:122) even when shuffle=False,
so the seed call placement above is load-bearing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Make megalodon importable from the LoQI repo source tree.
LOQI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LOQI_ROOT / "src"))
sys.path.insert(0, str(LOQI_ROOT / "app"))

import omegaconf  # noqa: E402  -- needed for ckpt unpickling
from omegaconf import OmegaConf  # noqa: E402
from rdkit import Chem  # noqa: E402

from megalodon.models.module import Graph3DInterpolantModel  # noqa: E402
from utils import generate_conformers_batch  # noqa: E402  app/utils.py


def _load_model(ckpt_path: Path, node_dist_path: Path):
    """Load LoQI checkpoint + minimal cfg compatible with generate_conformers_batch.

    Overrides ``sampling_params.node_distribution`` to point at a local copy of
    the train_n_h.pickle (the trainer's home dir is hardcoded in the saved
    hparams and does not exist on this machine).
    """
    # Peek hparams to read the original sampling_params, then override.
    raw_ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sp_saved = raw_ckpt["hyper_parameters"]["sampling_params"]
    sp_dict = OmegaConf.to_container(sp_saved, resolve=True) if isinstance(sp_saved, omegaconf.DictConfig) else dict(sp_saved)
    sp_dict["node_distribution"] = str(node_dist_path)
    sp_new = OmegaConf.create(sp_dict)

    model = Graph3DInterpolantModel.load_from_checkpoint(
        str(ckpt_path), sampling_params=sp_new,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Build a minimal cfg object with only the fields generate_conformers_batch reads.
    interp_dict = (
        OmegaConf.to_container(model.hparams.interpolant_params, resolve=True)
        if isinstance(model.hparams.interpolant_params, omegaconf.DictConfig)
        else dict(model.hparams.interpolant_params)
    )
    cfg = OmegaConf.create({
        "interpolant": interp_dict,
        "data": {
            "inference_batch_size": int(os.environ.get("LOQI_BATCH_SIZE", 32)),
            "batch_size": int(os.environ.get("LOQI_BATCH_SIZE", 32)),
            "aug_rotations": False,
            "scale_coords": 1.0,
        },
    })
    return model, cfg, device


def _write_sdf(mols: list, sdf_path: Path) -> int:
    """Write list of RDKit Mol objects to multi-conformer SDF.

    Returns the number of mol blocks written. Each mol carries one conformer.
    Caller is expected to loop the SDF reader to recover all conformers.
    """
    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with Chem.SDWriter(str(sdf_path)) as w:
        for mol in mols:
            if mol is None or mol.GetNumConformers() == 0:
                continue
            w.write(mol)
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path,
                   help="JSON file: list[{inchikey, smiles}].")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Directory to write <inchikey>.conformers.sdf into.")
    p.add_argument("--n_confs", type=int, default=10,
                   help="Conformers per SMILES (default 10).")
    p.add_argument("--ckpt", type=Path,
                   default=LOQI_ROOT / "data" / "loqi.ckpt",
                   help="LoQI checkpoint path.")
    p.add_argument("--node_dist", type=Path,
                   default=LOQI_ROOT / "data" / "chembl3d_stereo" /
                           "processed" / "train_n_h.pickle",
                   help="Local copy of the node-count distribution pickle.")
    p.add_argument(
        "--seed", type=int, default=0,
        help=("RNG seed for diffusion sampling. Default 0 for reproducibility. "
              "Stable default — downstream callers (aimnet2-solv, "
              "torchcosmors_benchmarks) rely on default RNG behavior. "
              "Any change to the default is a breaking API change."),
    )
    args = p.parse_args(argv)

    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if not args.input.is_file():
        print(f"ERROR: --input {args.input} not a file", file=sys.stderr)
        return 2
    if not args.ckpt.is_file():
        print(f"ERROR: --ckpt {args.ckpt} not found", file=sys.stderr)
        return 2
    if not args.node_dist.is_file():
        print(f"ERROR: --node_dist {args.node_dist} not found", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(args.input.read_text())
    if not isinstance(records, list):
        print("ERROR: --input must be a JSON list", file=sys.stderr)
        return 2

    print(f"[loqi_shim] loading {args.ckpt.name} ...", flush=True)
    t0 = time.perf_counter()
    model, cfg, device = _load_model(args.ckpt, args.node_dist)
    print(f"[loqi_shim] loaded in {time.perf_counter() - t0:.1f}s on {device}", flush=True)

    # Output sidecar: per-input record, success/fail status + sdf path.
    summary: list[dict] = []
    for rec in records:
        ik = rec.get("inchikey")
        smi = rec.get("smiles")
        if not (ik and smi):
            summary.append({"inchikey": ik, "smiles": smi, "ok": False,
                            "error": "missing inchikey or smiles"})
            continue
        sdf_path = args.output_dir / f"{ik}.conformers.sdf"
        t1 = time.perf_counter()
        try:
            mols, _, _, err = generate_conformers_batch(
                smi, model, cfg, n_confs=args.n_confs,
            )
        except Exception as e:
            summary.append({"inchikey": ik, "smiles": smi, "ok": False,
                            "error": f"generate raised: {e!r}"})
            print(f"[loqi_shim] {ik}: ERROR {e!r}", flush=True)
            continue
        if not mols:
            summary.append({"inchikey": ik, "smiles": smi, "ok": False,
                            "error": err or "empty result"})
            print(f"[loqi_shim] {ik}: empty (err={err!r})", flush=True)
            continue
        n_written = _write_sdf(mols, sdf_path)
        dt = time.perf_counter() - t1
        summary.append({
            "inchikey": ik, "smiles": smi, "ok": n_written > 0,
            "n_confs_written": n_written,
            "sdf": str(sdf_path),
            "walltime_s": round(dt, 2),
        })
        print(f"[loqi_shim] {ik}: {n_written} confs in {dt:.1f}s -> {sdf_path.name}", flush=True)

    # Write summary JSON next to the SDFs.
    summary_path = args.output_dir / "loqi_inference_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    n_ok = sum(1 for r in summary if r.get("ok"))
    print(f"\n[loqi_shim] done: {n_ok}/{len(summary)} records produced SDFs", flush=True)
    return 0 if n_ok == len(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
