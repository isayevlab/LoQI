"""
Compute conformer evaluation metrics between a target (generated) SDF and a reference
(ground-truth) SDF -- no model/checkpoint involved.

This is a fork of sample_conformers_processed.py that skips sampling entirely: instead of
generating conformers with a checkpoint, it loads already-generated molecules from --target
and pairs them with --reference, then runs the same ConformerEvaluationCallback used there.

Molecules are paired, in order of preference:
  1. by the --id_prop SD property (default "chemblid"), if every molecule in both files has it
     and it is unique among the references;
  2. by the RDKit "_Name" property, if every molecule in both files has one and it is unique
     among the references;
  3. by position, if the target holds k replicas of each reference in order (k = --n_confs, as
     written by sample_conformers_processed.py without shuffling). If "_Name" is set on both
     sides, every pair's names must agree.
Name pairing is skipped when reference names repeat, which happens for n-mers: both
sample_conformers_processed.py and convert_data_to_sdf.py set "_Name" to the SMILES, and
distinct dimers of the same compound share one.

Multi-fragment molecules (dimers, n-mers) are only evaluated when allow_fragments is on; it
comes from --allow_fragments, else evaluation.allow_fragments in --config, else is turned on
automatically if any reference molecule has more than one fragment.

Example:
    python scripts/compute_metrics_from_sdf.py \
        --target /data/.../csd_loqi_monomers_dih-relax_sample_ft.sdf \
        --reference /data/.../csd_loqi_monomers_dih-relax_reference.sdf \
        --config scripts/conf/loqi/loqi_finetune.yaml \
        --output /tmp/metrics.json
"""
import csv
import json
from argparse import ArgumentParser, BooleanOptionalAction
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


def pair_by_property(target_mols, reference_mols, prop):
    """Pair target/reference molecules by a property that is unique among the references."""
    reference_by_key = {m.GetProp(prop): m for m in reference_mols}
    paired_target, paired_reference, missing = [], [], []
    for m in target_mols:
        key = m.GetProp(prop)
        ref = reference_by_key.get(key)
        if ref is None:
            missing.append(key)
            continue
        paired_target.append(m)
        paired_reference.append(ref)
    if missing:
        print(f"WARNING: {len(missing)} target molecules had no matching reference by '{prop}' "
              f"(e.g. {missing[:5]}), dropped.")
    return paired_target, paired_reference


def pair_by_position(target_mols, reference_mols, n_confs):
    """Pair target i with reference i // k, where the target holds k replicas of each reference."""
    if n_confs is None:
        if len(target_mols) % len(reference_mols) != 0:
            raise ValueError(
                f"--target has {len(target_mols)} molecules, not a multiple of the "
                f"{len(reference_mols)} in --reference; cannot pair by position.")
        n_confs = len(target_mols) // len(reference_mols)
    elif len(target_mols) != n_confs * len(reference_mols):
        raise ValueError(
            f"--target has {len(target_mols)} molecules, expected --n_confs ({n_confs}) x "
            f"{len(reference_mols)} references; cannot pair by position.")

    paired_reference = [reference_mols[i // n_confs] for i in range(len(target_mols))]
    if all(m.HasProp("_Name") for m in target_mols + reference_mols):
        mismatched = [i for i, (t, r) in enumerate(zip(target_mols, paired_reference))
                      if t.GetProp("_Name") != r.GetProp("_Name")]
        if mismatched:
            raise ValueError(
                f"Positional pairing with {n_confs} conformer(s) per reference gives {len(mismatched)} "
                f"pairs whose '_Name' differs (first at target index {mismatched[0]}); the target "
                "order does not match the reference (e.g. it was sampled with --shuffle or "
                "--atom_aware_batching). Write a unique --id_prop into both files instead.")
    else:
        print("WARNING: pairing by position without '_Name' to verify the pairs.")
    print(f"Paired by position: {n_confs} target conformer(s) per reference molecule.")
    return list(target_mols), paired_reference


def pair_molecules(target_mols, reference_mols, id_prop="chemblid", n_confs=None):
    """Pair target/reference molecules by a unique property if possible, else by position."""
    for prop in (id_prop, "_Name"):
        if not prop or not all(m.HasProp(prop) for m in target_mols + reference_mols):
            continue
        n_unique = len({m.GetProp(prop) for m in reference_mols})
        if n_unique == len(reference_mols):
            print(f"Pairing target/reference molecules by '{prop}'.")
            return pair_by_property(target_mols, reference_mols, prop)
        print(f"'{prop}' is not unique in --reference ({len(reference_mols) - n_unique} repeats, "
              "e.g. n-mers of the same compound sharing a SMILES); not pairing by it.")
    return pair_by_position(target_mols, reference_mols, n_confs)


def check_atoms_match(generated, references):
    """Drop pairs whose atoms (element, in order) differ, which would make every metric meaningless."""
    kept_gen, kept_ref, n_dropped = [], [], 0
    for gen, ref in zip(generated, references):
        if [a.GetAtomicNum() for a in gen.GetAtoms()] != [a.GetAtomicNum() for a in ref.GetAtoms()]:
            n_dropped += 1
            continue
        kept_gen.append(gen)
        kept_ref.append(ref)
    if n_dropped:
        print(f"WARNING: {n_dropped} pairs have different atoms in target and reference, dropped.")
    return kept_gen, kept_ref


def resolve_allow_fragments(cli_value, cfg, reference_mols):
    """--allow_fragments, else evaluation.allow_fragments in the config, else auto-detect."""
    if cli_value is not None:
        return cli_value
    cfg_value = OmegaConf.select(cfg, "evaluation.allow_fragments", default=None)
    if cfg_value is not None:
        return bool(cfg_value)
    n_multi = sum(1 for m in reference_mols if len(Chem.GetMolFrags(m)) > 1)
    if n_multi:
        print(f"{n_multi} reference molecules have more than one fragment; enabling allow_fragments.")
    return n_multi > 0


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
    parser.add_argument("--id_prop", type=str, default="chemblid",
                        help="SD property holding a unique molecule ID to pair target/reference by.")
    parser.add_argument("--n_confs", type=int, default=None,
                        help="Conformers per reference in --target, for pairing by position "
                             "(default: inferred from the file sizes).")
    parser.add_argument("--allow_fragments", action=BooleanOptionalAction, default=None,
                        help="Evaluate multi-fragment molecules (dimers, n-mers). Default: "
                             "evaluation.allow_fragments in --config, else on if any reference "
                             "molecule has more than one fragment.")
    parser.add_argument("--opt", action=BooleanOptionalAction, default=None,
                        help="Run AIMNet2 optimization and the opt_* metrics. Default: "
                             "evaluation.energy_metrics_args.opt_metrics in --config.")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    target_mols = load_sdf_molecules(args.target)
    reference_mols = load_sdf_molecules(args.reference)
    if not target_mols:
        raise ValueError(f"No valid molecules found in --target: {args.target}")
    if not reference_mols:
        raise ValueError(f"No valid molecules found in --reference: {args.reference}")

    generated, references = pair_molecules(target_mols, reference_mols,
                                           id_prop=args.id_prop, n_confs=args.n_confs)
    generated, references = check_atoms_match(generated, references)
    allow_fragments = resolve_allow_fragments(args.allow_fragments, cfg, references)
    print(f"allow_fragments: {allow_fragments}")

    for gen, ref in tqdm(zip(generated, references), total=len(generated), desc="Filling missing reference conformers"):
        if ref.GetNumConformers() == 0:
            ref.AddConformer(Chem.Conformer(ref.GetNumAtoms()))
            conf = gen.GetConformer(0)
            pos = conf.GetPositions()
            conf.SetPositions(pos)
            ref.AddConformer(conf)

    energy_metrics_args = OmegaConf.to_container(cfg.evaluation.energy_metrics_args, resolve=True)
    if args.opt is not None:
        energy_metrics_args["opt_metrics"] = args.opt
    print(f"optimization: {energy_metrics_args['opt_metrics']}")

    processed_stats_dir = f"{cfg.data.dataset_root}/processed"
    stats = Statistics.load_statistics(processed_stats_dir, "train")
    eval_cb = ConformerEvaluationCallback(
        compute_3D_metrics=cfg.evaluation.compute_3D_metrics,
        compute_energy_metrics=cfg.evaluation.compute_energy_metrics,
        energy_metrics_args=energy_metrics_args,
        statistics=stats,
        scale_coords=cfg.evaluation.scale_coords,
        compute_stereo_metrics=True,
        allow_fragments=allow_fragments,
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
