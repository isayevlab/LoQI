"""Sample conformers with a LoQI checkpoint.

Thin wrapper around the ``loqi`` package: validation, featurisation, batching and sampling live in
``loqi.featurize`` / ``loqi.api``. The optional AIMNet2 optimisation, iRMSD pruning and the
``--eval`` metrics are script-only postprocessing steps.
"""

import os
import pickle
from argparse import ArgumentParser, BooleanOptionalAction
from importlib.resources import files

import numpy as np
import torch
from omegaconf import OmegaConf
from rdkit import Chem
from tqdm import tqdm

from loqi.api import iter_sampled_batches, load_model
from loqi.featurize import build_sampling_loader, load_molecules, mols_to_data_list
from megalodon.data.statistics import Statistics
from megalodon.metrics.conformer_evaluation_callback import (
    ConformerEvaluationCallback,
    write_coords_to_mol,
)
from megalodon.metrics.molecule_metrics_aimnet2 import MoleculeAIMNet2Metrics

Chem.SetUseLegacyStereoPerception(True)

BUNDLED_AIMNET2_MODEL = files("megalodon").joinpath("metrics/aimnet2/cpcm_model/wb97m_cpcms_v2_0.jpt")


def optimize_with_aimnet(
    molecules,
    cfg,
    opt_batch_size=None,
    fmax=0.05,
    max_nstep=250,
):
    aimnet_path = cfg.evaluation.energy_metrics_args.model_path or str(BUNDLED_AIMNET2_MODEL)
    if not os.path.exists(str(aimnet_path)):
        return None, None, f"AIMNet2 model not found: {aimnet_path}"
    if opt_batch_size is None:
        opt_batch_size = int(getattr(cfg.evaluation.energy_metrics_args, "batchsize", 100))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_metrics = MoleculeAIMNet2Metrics(
        model_path=str(aimnet_path),
        batchsize=int(opt_batch_size),
        opt_metrics=True,
        opt_params={"fmax": float(fmax), "max_nstep": int(max_nstep)},
        device=device,
    )
    try:
        _, _, opt_mols, opt_energies = energy_metrics(molecules, reference_molecules=None, return_molecules=True)
        return opt_mols, opt_energies, None
    except Exception as exc:  # noqa: BLE001 - reported to the caller as an error string
        return None, None, f"Optimization failed: {exc}"


def select_unique_with_irmsd(molecules, rthr=0.125):
    if molecules is None:
        return None, None, "No molecules provided for iRMSD pruning."
    if len(molecules) == 0:
        return [], [], None
    if len(molecules) == 1:
        # Nothing to prune for a single conformer.
        return molecules, [0], None

    try:
        from irmsd import sorter_irmsd_rdkit  # type: ignore
    except Exception:  # noqa: BLE001
        return None, None, "iRMSD is not installed. Install with: pip install irmsd"
    try:
        # iinversion=2 disables inversion.
        groups, _ = sorter_irmsd_rdkit(molecules, rthr=float(rthr), iinversion=2, allcanon=True, printlvl=0)
        groups = np.asarray(groups).reshape(-1)
        if groups.shape[0] != len(molecules):
            return None, None, f"iRMSD returned unexpected group shape {groups.shape}; expected ({len(molecules)},)."
        selected_indices = []
        seen = set()
        for idx, gid in enumerate(groups.tolist()):
            if gid not in seen:
                seen.add(gid)
                selected_indices.append(idx)
        if not selected_indices:
            return None, None, "iRMSD did not produce any unique representatives."
        unique_mols = [molecules[i] for i in selected_indices]
        return unique_mols, selected_indices, None
    except Exception as exc:  # noqa: BLE001
        return None, None, f"iRMSD pruning failed: {exc}"


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config YAML. Defaults to the inference config bundled with the loqi package for the checkpoint.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="loqi",
        help="Checkpoint path or registered model name ('loqi', 'loqi_flow'; downloaded on first use).",
    )
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_confs", type=int, default=1)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Sampling batch size. If omitted, uses data.inference_batch_size from config.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=25,
        help=(
            "Sampling steps (default: 25). Diffusion models were trained with 25 steps and "
            "are not expected to work well for other values. Flow-matching models can be "
            "run with different step counts."
        ),
    )
    parser.add_argument(
        "--add-hs",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "Add hydrogens during SMILES validation/featurization. "
            "Use --no-add-hs when input already contains explicit hydrogens."
        ),
    )
    parser.add_argument("--eval", action="store_true", help="Run evaluation (off by default)")
    parser.add_argument(
        "--postprocess",
        choices=["none", "optimization", "optimization+irmsd"],
        default="none",
        help="Optional postprocessing of generated conformers.",
    )
    parser.add_argument(
        "--optimization_batch_size",
        type=int,
        default=None,
        help="Batch size for AIMNet2 optimization (default: cfg.evaluation.energy_metrics_args.batchsize).",
    )
    parser.add_argument(
        "--opt_fmax",
        type=float,
        default=None,
        help="Optimization force threshold (default: cfg.evaluation.energy_metrics_args.opt_params.fmax).",
    )
    parser.add_argument(
        "--opt_max_nstep",
        type=int,
        default=None,
        help="Maximum optimization steps (default: cfg.evaluation.energy_metrics_args.opt_params.max_nstep).",
    )
    parser.add_argument("--irmsd_rthr", type=float, default=0.125, help="iRMSD pruning threshold.")
    parser.add_argument(
        "--atom-aware-batching",
        action=BooleanOptionalAction,
        default=True,
        help="Enable atom-aware batching with AdaptiveBatchSampler. Use --no-atom-aware-batching to disable.",
    )
    parser.add_argument(
        "--target-molecule-size",
        type=int,
        default=50,
        help="Target molecule size for atom-aware batching (default: 50).",
    )
    parser.add_argument(
        "--shuffle",
        action=BooleanOptionalAction,
        default=False,
        help="Shuffle conformer replicas before batching. Use --no-shuffle to disable.",
    )
    parser.add_argument(
        "--use-stereo-bonds",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "Add stereochemistry-derived graph edges during featurization. "
            "Use --no-use-stereo-bonds for ablation experiments."
        ),
    )
    args = parser.parse_args()

    # Load model
    loaded = load_model(args.ckpt, config=args.config)
    model, cfg = loaded.model, loaded.config
    cfg_opt_params = getattr(getattr(cfg.evaluation, "energy_metrics_args", None), "opt_params", None)
    opt_fmax = float(args.opt_fmax) if args.opt_fmax is not None else float(getattr(cfg_opt_params, "fmax", 0.05))
    opt_max_nstep = (
        int(args.opt_max_nstep) if args.opt_max_nstep is not None else int(getattr(cfg_opt_params, "max_nstep", 250))
    )
    sample_batch_size = args.batch_size if args.batch_size is not None else loaded.default_batch_size

    # Load molecules and replicate them n_confs times.
    # Use provided 3D coordinates only for SDF inputs that already contain conformers.
    input_is_sdf = os.path.isfile(args.input) and args.input.endswith(".sdf")
    mols, validation_errors = load_molecules(args.input, add_hs=args.add_hs)
    for err in validation_errors:
        print(f"WARNING: {err}")
    if not mols:
        raise ValueError("No valid molecules left after validation/revalidation checks.")
    has_3d_input = any(mol.GetNumConformers() > 0 for mol in mols) if input_is_sdf else False
    use_3d_input = input_is_sdf and has_3d_input
    data_list = mols_to_data_list(
        mols,
        n_confs=args.n_confs,
        use_3d_input=use_3d_input,
        use_stereo_bonds=bool(args.use_stereo_bonds),
    )
    loader = build_sampling_loader(
        data_list,
        sample_batch_size,
        atom_aware_batching=bool(args.atom_aware_batching),
        shuffle=bool(args.shuffle),
        target_molecule_size=int(args.target_molecule_size),
    )

    # Sampling
    generated = []
    skip_eval = not args.eval
    references = [] if not skip_eval else None
    ids = []
    timesteps = args.n_steps

    for batch, coords_list in tqdm(iter_sampled_batches(loaded, loader, steps=timesteps), desc="Sampling"):
        mols_gen = [write_coords_to_mol(mol, coords) for mol, coords in zip(batch["mol"], coords_list, strict=True)]
        generated.extend(mols_gen)
        if not skip_eval:
            references.extend(batch["mol"])
        ids.extend([m.GetProp("_Name") if m.HasProp("_Name") else "NA" for m in batch["mol"]])

    energies = None
    if args.postprocess in {"optimization", "optimization+irmsd"}:
        optimized, energies, opt_error = optimize_with_aimnet(
            generated,
            cfg,
            opt_batch_size=args.optimization_batch_size,
            fmax=opt_fmax,
            max_nstep=opt_max_nstep,
        )
        if opt_error is not None:
            raise RuntimeError(opt_error)
        generated = optimized
        if hasattr(energies, "detach"):
            energies = energies.detach().cpu().numpy()
        else:
            energies = np.asarray(energies)
        print(f"Optimization complete: {len(generated)} conformers.")

    if args.postprocess == "optimization+irmsd":
        unique_mols, selected_indices, irmsd_error = select_unique_with_irmsd(generated, rthr=args.irmsd_rthr)
        if irmsd_error is not None:
            raise RuntimeError(irmsd_error)
        generated = unique_mols
        ids = [ids[i] for i in selected_indices]
        if references is not None:
            references = [references[i] for i in selected_indices]
        if energies is not None:
            energies = energies[selected_indices]
        print(f"iRMSD unique selection complete: {len(generated)} conformers.")

    # Save output
    if args.output.endswith(".sdf"):
        from rdkit.Chem import SDWriter

        writer = SDWriter(args.output)
        ev2kcalpermol = 23.060547830619026
        for idx, mol in enumerate(generated):
            if energies is not None:
                mol.SetProp("Energy_kcal_mol", f"{float(energies[idx]) * ev2kcalpermol:.6f}")
            writer.write(mol)
        writer.close()
    else:
        output_dict = {"generated": generated, "ids": ids}
        if energies is not None:
            output_dict["energies"] = energies
        if references is not None:
            output_dict["reference"] = references
        with open(args.output, "wb") as f:
            pickle.dump(output_dict, f)

    # Evaluate only if references are available and evaluation is not skipped
    if not skip_eval and references and has_3d_input:
        if not cfg.data.get("dataset_root"):
            raise ValueError("--eval needs a config with data.dataset_root (e.g. scripts/conf/loqi/loqi.yaml).")
        stats = Statistics.load_statistics(cfg.data.dataset_root + "/processed", "train")
        eval_cb = ConformerEvaluationCallback(
            timesteps=timesteps,
            compute_3D_metrics=cfg.evaluation.compute_3D_metrics,
            compute_energy_metrics=cfg.evaluation.compute_energy_metrics,
            energy_metrics_args=OmegaConf.to_container(cfg.evaluation.energy_metrics_args, resolve=True),
            statistics=stats,
            scale_coords=cfg.evaluation.scale_coords,
            compute_stereo_metrics=True,
        )
        for gen, ref in zip(generated, references, strict=True):
            if ref.GetNumConformers() == 0:
                ref.AddConformer(Chem.Conformer(ref.GetNumAtoms()))
                conf = gen.GetConformer(0)
                pos = conf.GetPositions()
                conf.SetPositions(pos)
                ref.AddConformer(conf)
        results = eval_cb.evaluate_molecules(generated, reference_molecules=references, device=model.device)
        print("Evaluation Results:")
        print(results)

    print(f"Generated {len(generated)} conformers for {len(set(ids))} unique molecules.")


if __name__ == "__main__":
    main()
