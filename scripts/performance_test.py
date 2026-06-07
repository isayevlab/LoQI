import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
import omegaconf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from rdkit import Chem
from rdkit.Chem import SDWriter

import sample_conformers as sc

from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.data.molecule_dataset import MoleculeDataset
from megalodon.metrics.conformer_evaluation_callback import convert_coords_to_np, write_coords_to_mol
from megalodon.models.module import Graph3DInterpolantModel


def parse_sizes(sizes: str):
    return [int(x.strip()) for x in sizes.split(",") if x.strip()]


def resolve_dataset_args(dataset_pt: Path):
    processed_folder = dataset_pt.parent.name
    dataset_root = dataset_pt.parent.parent
    split = dataset_pt.name.split("_")[0]
    return dataset_root, processed_folder, split


def select_indices_by_size(dataset, sizes, n_per_size):
    slices_x = dataset.slices["x"]
    atom_counts = (slices_x[1:] - slices_x[:-1]).cpu().numpy().astype(int)
    grouped = {size: np.where(atom_counts == size)[0].tolist() for size in sizes}
    selected = {}
    for size in sizes:
        candidates = grouped[size]
        if len(candidates) == 0:
            selected[size] = []
            continue
        if len(candidates) < n_per_size:
            selected[size] = candidates
            print(
                f"WARNING: size={size} has only {len(candidates)} molecules "
                f"(requested {n_per_size}); using all available."
            )
        else:
            # Deterministic selection: keep the first N in dataset order.
            selected[size] = candidates[:n_per_size]
    return selected


def write_sdf_set(dataset, selected_indices, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir / "selected_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["size", "rank", "dataset_idx", "chemblid", "smiles", "sdf_path"],
        )
        writer.writeheader()

        for size, indices in selected_indices.items():
            size_dir = outdir / f"size_{size}"
            size_dir.mkdir(parents=True, exist_ok=True)
            combined_writer = SDWriter(str(outdir / f"size_{size}_selected.sdf"))

            for rank, idx in enumerate(indices):
                data = dataset[int(idx)]
                mol = Chem.Mol(data.mol)
                if mol.GetNumConformers() == 0:
                    raise ValueError(f"Selected molecule at index {idx} has no conformer.")

                chemblid = data.chemblid if hasattr(data, "chemblid") else ""
                smiles = data.smiles if hasattr(data, "smiles") else Chem.MolToSmiles(mol)
                mol.SetProp("_Name", str(chemblid))
                mol.SetProp("dataset_idx", str(int(idx)))
                mol.SetProp("atom_count", str(size))

                mol_file = size_dir / f"mol_{rank:04d}_idx_{int(idx)}.sdf"
                per_writer = SDWriter(str(mol_file))
                per_writer.write(mol)
                per_writer.close()
                combined_writer.write(mol)

                writer.writerow(
                    {
                        "size": size,
                        "rank": rank,
                        "dataset_idx": int(idx),
                        "chemblid": chemblid,
                        "smiles": smiles,
                        "sdf_path": str(mol_file),
                    }
                )
            combined_writer.close()
    return manifest_path


def load_model(cfg_path: Path, ckpt_path: Path, n_steps: int):
    torch.serialization.add_safe_globals([DictConfig, ListConfig])
    cfg = OmegaConf.load(str(cfg_path))
    model = Graph3DInterpolantModel.load_from_checkpoint(
        str(ckpt_path),
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preporcessor=BatchPreProcessor(cfg.data.aug_rotations, cfg.data.scale_coords),
        weights_only=False,
    )
    cfg.interpolant.timesteps = int(n_steps)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), cfg


def load_single_sdf(path: str):
    suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
    mols = [m for m in suppl if m is not None]
    if len(mols) != 1:
        raise ValueError(f"Expected exactly one molecule in {path}, found {len(mols)}")
    return mols[0]


def benchmark_manifest(
    manifest_path: Path,
    model,
    cfg,
    out_csv: Path,
    n_confs: int,
    gen_batch_size: int,
    atom_aware_batching: bool,
    target_molecule_size: int,
    shuffle: bool,
    run_optimization: bool,
    optimization_batch_size: int,
    opt_fmax: float,
    opt_max_nstep: int,
    irmsd_after_opt: bool,
    irmsd_rthr: float,
):
    with open(manifest_path, newline="") as fin, open(out_csv, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        fieldnames = [
            "size",
            "rank",
            "dataset_idx",
            "chemblid",
            "sdf_path",
            "n_confs",
            "gen_time_s",
            "gen_time_per_conf_s",
            "opt_time_s",
            "opt_time_per_conf_s",
            "n_generated",
            "n_after_postprocess",
            "status",
            "error",
        ]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            record = {k: row.get(k, "") for k in ["size", "rank", "dataset_idx", "chemblid", "sdf_path"]}
            record.update(
                {
                    "n_confs": n_confs,
                    "gen_time_s": "",
                    "gen_time_per_conf_s": "",
                    "opt_time_s": "",
                    "opt_time_per_conf_s": "",
                    "n_generated": 0,
                    "n_after_postprocess": 0,
                    "status": "ok",
                    "error": "",
                }
            )
            try:
                mol = load_single_sdf(row["sdf_path"])
                data_list = sc.mols_to_data_list([mol], n_confs=n_confs, use_3d_input=True)
                loader = sc.build_sampling_loader(
                    data_list=data_list,
                    sample_batch_size=gen_batch_size,
                    atom_aware_batching=atom_aware_batching,
                    shuffle=shuffle,
                    target_molecule_size=target_molecule_size,
                )

                generated = []
                t0 = time.perf_counter()
                for batch in loader:
                    batch = batch.to(model.device)
                    sample = model.sample(batch=batch, timesteps=cfg.interpolant.timesteps, pre_format=True)
                    coords_list = convert_coords_to_np(sample)
                    generated.extend(
                        [write_coords_to_mol(m, c) for m, c in zip(batch["mol"], coords_list)]
                    )
                gen_time = time.perf_counter() - t0
                record["gen_time_s"] = f"{gen_time:.6f}"
                record["gen_time_per_conf_s"] = f"{gen_time / max(len(generated), 1):.6f}"
                record["n_generated"] = len(generated)
                post = generated

                if run_optimization:
                    t1 = time.perf_counter()
                    post, _, opt_err = sc.optimize_with_aimnet(
                        generated,
                        cfg,
                        opt_batch_size=optimization_batch_size,
                        fmax=opt_fmax,
                        max_nstep=opt_max_nstep,
                    )
                    if opt_err is not None:
                        raise RuntimeError(opt_err)
                    if irmsd_after_opt:
                        uniq, _, irmsd_err = sc.select_unique_with_irmsd(post, rthr=irmsd_rthr)
                        if irmsd_err is not None:
                            raise RuntimeError(irmsd_err)
                        post = uniq
                    opt_time = time.perf_counter() - t1
                    record["opt_time_s"] = f"{opt_time:.6f}"
                    record["opt_time_per_conf_s"] = f"{opt_time / max(len(generated), 1):.6f}"

                record["n_after_postprocess"] = len(post)
            except Exception as exc:
                record["status"] = "error"
                record["error"] = str(exc)

            writer.writerow(record)
            fout.flush()


def resolve_optimization_settings(cfg, args):
    energy_args = getattr(cfg.evaluation, "energy_metrics_args", None)
    cfg_batchsize = 100
    cfg_fmax = 0.05
    cfg_max_nstep = 250

    if energy_args is not None:
        cfg_batchsize = int(getattr(energy_args, "batchsize", cfg_batchsize))
        opt_params = getattr(energy_args, "opt_params", None)
        if opt_params is not None:
            cfg_fmax = float(getattr(opt_params, "fmax", cfg_fmax))
            cfg_max_nstep = int(getattr(opt_params, "max_nstep", cfg_max_nstep))

    optimization_batch_size = (
        int(args.optimization_batch_size)
        if args.optimization_batch_size is not None
        else cfg_batchsize
    )
    opt_fmax = float(args.opt_fmax) if args.opt_fmax is not None else cfg_fmax
    opt_max_nstep = int(args.opt_max_nstep) if args.opt_max_nstep is not None else cfg_max_nstep
    return optimization_batch_size, opt_fmax, opt_max_nstep


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare fixed-size SDF sets from test_h.pt and benchmark per-molecule "
            "generation/optimization timings."
        )
    )
    parser.add_argument(
        "--dataset_pt",
        type=Path,
        default=Path("data/chembl3d_stereo/processed/train_h.pt"),
    )
    parser.add_argument("--sizes", type=str, default="10,25,50,100")
    parser.add_argument("--n_per_size", type=int, default=1000)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/performance_test"))
    parser.add_argument("--prepare_only", action="store_true")

    parser.add_argument("--config", type=Path, default=Path("scripts/conf/loqi/loqi.yaml"))
    parser.add_argument("--ckpt", type=Path, default=Path("data/loqi.ckpt"))
    parser.add_argument("--n_steps", type=int, default=25)
    parser.add_argument("--n_confs", type=int, default=100)
    parser.add_argument("--generation_batch_size", type=int, default=1)
    parser.add_argument("--atom_aware_batching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--target_molecule_size", type=int, default=50)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--run_optimization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimization_batch_size", type=int, default=None)
    parser.add_argument("--opt_fmax", type=float, default=None)
    parser.add_argument("--opt_max_nstep", type=int, default=None)
    parser.add_argument("--irmsd_after_opt", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--irmsd_rthr", type=float, default=0.125)
    args = parser.parse_args()

    dataset_root, processed_folder, split = resolve_dataset_args(args.dataset_pt)
    dataset = MoleculeDataset(
        root=str(dataset_root), processed_folder=str(processed_folder), split=split
    )

    sizes = parse_sizes(args.sizes)
    selected = select_indices_by_size(dataset, sizes, args.n_per_size)
    manifest_path = write_sdf_set(dataset, selected, args.outdir)
    print(f"Prepared SDF set. Manifest: {manifest_path}")

    if args.prepare_only:
        return

    model, cfg = load_model(args.config, args.ckpt, args.n_steps)
    optimization_batch_size, opt_fmax, opt_max_nstep = resolve_optimization_settings(cfg, args)

    out_csv = args.outdir / "timings_per_molecule.csv"
    benchmark_manifest(
        manifest_path=manifest_path,
        model=model,
        cfg=cfg,
        out_csv=out_csv,
        n_confs=args.n_confs,
        gen_batch_size=args.generation_batch_size,
        atom_aware_batching=bool(args.atom_aware_batching),
        target_molecule_size=int(args.target_molecule_size),
        shuffle=bool(args.shuffle),
        run_optimization=bool(args.run_optimization),
        optimization_batch_size=optimization_batch_size,
        opt_fmax=opt_fmax,
        opt_max_nstep=opt_max_nstep,
        irmsd_after_opt=bool(args.irmsd_after_opt),
        irmsd_rthr=float(args.irmsd_rthr),
    )
    print(f"Benchmark complete. Timings: {out_csv}")


if __name__ == "__main__":
    main()
