"""Command line interface: ``loqi sample`` and ``loqi download``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loqi.registry import MODELS, checkpoint_path, default_cache_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="loqi", description="LoQI molecular conformer generation.")
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="Generate conformers for SMILES and write them to an SDF file.")
    sample.add_argument("--smiles", action="append", default=[], metavar="SMILES", help="SMILES string; repeatable.")
    sample.add_argument(
        "--input", type=Path, help="Text file with one SMILES per line (first whitespace-separated token is used)."
    )
    sample.add_argument("--n-confs", type=int, default=10, help="Conformers per molecule (default: 10).")
    sample.add_argument("--output", type=Path, required=True, help="Output SDF file (one record per conformer).")
    sample.add_argument(
        "--model", default="loqi", help=f"Registered model ({', '.join(MODELS)}) or checkpoint path (default: loqi)."
    )
    sample.add_argument(
        "--config", default=None, help="Bundled config name or YAML path; needed for checkpoints outside the registry."
    )
    sample.add_argument("--device", default=None, help="torch device (default: cuda if available, else cpu).")
    sample.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    sample.add_argument("--steps", type=int, default=None, help="Sampling steps (default: the model's training value).")
    sample.add_argument(
        "--batch-atoms", type=int, default=None, help="Atom budget per batch at the 50-atom reference size."
    )
    sample.add_argument(
        "--no-add-hs", dest="add_hs", action="store_false", help="Do not add hydrogens to the input SMILES."
    )

    download = sub.add_parser("download", help="Download a registered checkpoint into the cache and print its path.")
    download.add_argument("--model", default="loqi", help=f"Registered model ({', '.join(MODELS)}); default: loqi.")
    download.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"Cache directory (default: $LOQI_CACHE_DIR or {default_cache_dir()}).",
    )
    return parser


def _read_smiles_file(path: Path) -> list[str]:
    smiles = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                smiles.append(line.split()[0])
    return smiles


def _run_sample(args: argparse.Namespace) -> int:
    from rdkit import Chem

    from loqi.api import generate_conformers, load_model
    from loqi.featurize import legacy_stereo_perception, prepare_molecule

    smiles = list(args.smiles)
    if args.input is not None:
        smiles.extend(_read_smiles_file(args.input))
    if not smiles:
        print("error: provide at least one --smiles or an --input file", file=sys.stderr)
        return 2

    valid = []
    with legacy_stereo_perception():
        for smi in smiles:
            try:
                prepare_molecule(smi, add_hs=args.add_hs)
            except ValueError as exc:
                print(f"skipping: {exc}", file=sys.stderr)
                continue
            valid.append(smi)
    if not valid:
        print("error: no valid SMILES", file=sys.stderr)
        return 1

    loaded = load_model(args.model, device=args.device, config=args.config)
    mols = generate_conformers(
        valid,
        args.n_confs,
        model=loaded,
        seed=args.seed,
        steps=args.steps,
        add_hs=args.add_hs,
        batch_atoms=args.batch_atoms,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with Chem.SDWriter(str(args.output)) as writer:
        for smi, mol in zip(valid, mols, strict=True):
            mol.SetProp("_Name", smi)
            mol.SetProp("loqi_model", loaded.name)
            for conf in mol.GetConformers():
                mol.SetIntProp("loqi_conformer_id", conf.GetId())
                writer.write(mol, confId=conf.GetId())
                n_written += 1
    n_failed = sum(mol.GetIntProp("loqi_failed") for mol in mols)
    print(
        f"wrote {n_written} conformers for {len(mols)} molecules to {args.output}"
        + (f" ({n_failed} samples failed)" if n_failed else ""),
        file=sys.stderr,
    )
    return 0


def _run_download(args: argparse.Namespace) -> int:
    if args.model not in MODELS:
        print(f"error: unknown model {args.model!r}; choose from {', '.join(MODELS)}", file=sys.stderr)
        return 2
    print(checkpoint_path(args.model, cache_dir=args.cache_dir))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "sample":
        return _run_sample(args)
    if args.command == "download":
        return _run_download(args)
    return 2  # pragma: no cover - argparse enforces the subcommand


if __name__ == "__main__":
    sys.exit(main())
