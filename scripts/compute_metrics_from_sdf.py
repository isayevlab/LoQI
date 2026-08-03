"""
Compute conformer evaluation metrics between a target (generated) SDF and a reference
(ground-truth) SDF -- no model/checkpoint involved.

This is a fork of sample_conformers_processed.py that skips sampling entirely: instead of
generating conformers with a checkpoint, it loads already-generated molecules from --target
and pairs them with --reference, then runs the same ConformerEvaluationCallback used there.

Molecules are paired by the RDKit "_Name" property when every molecule in both files has one
set (e.g. outputs from sample_conformers_processed.py, which preserves it); otherwise they are
paired by position, requiring both files to have the same number of molecules.

Example:
    python scripts/compute_metrics_from_sdf.py \
        --target /data/.../csd_loqi_monomers_dih-relax_sample_ft.sdf \
        --reference /data/.../csd_loqi_monomers_dih-relax_reference.sdf \
        --config scripts/conf/loqi/loqi_finetune.yaml \
        --output /tmp/metrics.json
"""
import csv
import json
from argparse import ArgumentParser
from rdkit import Chem
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from megalodon.data.statistics import Statistics
from megalodon.metrics.conformer_evaluation_callback import ConformerEvaluationCallback

Chem.SetUseLegacyStereoPerception(True)


def load_sdf_molecules(path):
    suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
    mols = []
    for idx, mol in enumerate(tqdm(suppl, desc=f"Loading {path}")):
        if mol is None:
            print(f"WARNING: {path}: entry {idx}: RDKit failed to read molecule, skipping.")
            continue
        mols.append(mol)
    return mols


def pair_molecules(target_mols, reference_mols):
    """Pair target/reference molecules by '_Name' if all entries have one set, else by position."""
    target_named = all(m.HasProp("_Name") for m in target_mols)
    reference_named = all(m.HasProp("_Name") for m in reference_mols)

    if target_named and reference_named:
        reference_by_name = {}
        for m in reference_mols:
            name = m.GetProp("_Name")
            reference_by_name.setdefault(name, m)
        paired_target, paired_reference, missing = [], [], []
        for m in target_mols:
            name = m.GetProp("_Name")
            ref = reference_by_name.get(name)
            if ref is None:
                missing.append(name)
                continue
            paired_target.append(m)
            paired_reference.append(ref)
        if missing:
            print(f"WARNING: {len(missing)} target molecules had no matching reference by name "
                  f"(e.g. {missing[:5]}), dropped.")
        return paired_target, paired_reference

    if len(target_mols) != len(reference_mols):
        raise ValueError(
            f"--target has {len(target_mols)} molecules and --reference has {len(reference_mols)}; "
            "cannot pair by position. Set '_Name' on every molecule in both files to pair by name instead."
        )
    print("WARNING: pairing target/reference molecules by position "
          "('_Name' not set on every molecule in both files).")
    return list(target_mols), list(reference_mols)


def main():
    parser = ArgumentParser()
    parser.add_argument("--target", type=str, required=True, help="SDF of generated conformers.")
    parser.add_argument("--reference", type=str, required=True, help="SDF of reference/ground-truth conformers.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default=None, help="Optional path to save results (.json).")
    parser.add_argument("--output_sdf", type=str, default=None,
                         help="Optional path to save the AIMNet2-optimized generated structures "
                              "(requires compute_energy_metrics and opt_metrics enabled in --config).")
    parser.add_argument("--output_log", type=str, default=None,
                         help="Optional path to save a per-molecule optimization log (.csv) with "
                              "smiles, reference/pre/post-optimization energy, whether topology "
                              "was preserved, and R/S and E/Z stereocenter correctness counts.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    target_mols = load_sdf_molecules(args.target)
    reference_mols = load_sdf_molecules(args.reference)
    if not target_mols:
        raise ValueError(f"No valid molecules found in --target: {args.target}")
    if not reference_mols:
        raise ValueError(f"No valid molecules found in --reference: {args.reference}")

    generated, references = pair_molecules(target_mols, reference_mols)

    for gen, ref in tqdm(zip(generated, references), total=len(generated), desc="Filling missing reference conformers"):
        if ref.GetNumConformers() == 0:
            ref.AddConformer(Chem.Conformer(ref.GetNumAtoms()))
            conf = gen.GetConformer(0)
            pos = conf.GetPositions()
            conf.SetPositions(pos)
            ref.AddConformer(conf)

    processed_stats_dir = f"{cfg.data.dataset_root}/processed"
    stats = Statistics.load_statistics(processed_stats_dir, "train")
    eval_cb = ConformerEvaluationCallback(
        compute_3D_metrics=cfg.evaluation.compute_3D_metrics,
        compute_energy_metrics=cfg.evaluation.compute_energy_metrics,
        energy_metrics_args=OmegaConf.to_container(cfg.evaluation.energy_metrics_args, resolve=True),
        statistics=stats,
        scale_coords=cfg.evaluation.scale_coords,
        compute_stereo_metrics=True,
    )
    results = eval_cb.evaluate_molecules(
        generated, reference_molecules=references, device=device,
        return_optimized_molecules=args.output_sdf is not None,
        return_optimization_log=args.output_log is not None)

    optimized_molecules = results.pop("optimized_molecules", None)
    optimization_log = results.pop("optimization_log", None)

    print(f"Evaluated {len(generated)} molecule pairs.")
    print("Evaluation Results:")
    print(results)

    if args.output_sdf is not None:
        if optimized_molecules is None:
            print(f"WARNING: --output_sdf was set but no optimized structures were produced; "
                  f"skipping write to {args.output_sdf}.")
        else:
            writer = Chem.SDWriter(args.output_sdf)
            for mol in optimized_molecules:
                writer.write(mol)
            writer.close()
            print(f"Saved {len(optimized_molecules)} optimized structures to {args.output_sdf}")

    if args.output_log is not None:
        fieldnames = ["smiles", "reference_energy", "energy_before_opt", "energy_after_opt",
                      "topology_preserved", "rs_correct", "rs_total", "ez_correct", "ez_total"]
        with open(args.output_log, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(optimization_log)
        print(f"Saved optimization log for {len(optimization_log)} molecules to {args.output_log}")

    if args.output is not None:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
