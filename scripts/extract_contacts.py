"""
Extract intermolecular contacts from an n-mer SDF's ATOM_CONTACTS tag (written by
csd_extract_dimers.py / fix_contacts.py) into a JSON file used by the constrained optimization
in validation (evaluation.energy_metrics_args.constrained_opt_params.contacts_file):
    {"<SDF title>": [[i, j], ...], ...}
with 0-based heavy-atom index pairs (a hydrogen side resolved to its bonded heavy atom). The
SDF title is the processed dataset's `chemblid`, and the atom order is the same.

Example:
    python scripts/extract_contacts.py \
        --sdf /data/jarret/loqi/csd_loqi_data/dimers/csd_loqi_dimers_20k__dih-relax_k10__FIXED1.sdf \
        --output /data/jarret/loqi/csd_loqi_data/dimers/csd_loqi_dimers_20k__dih-relax/contacts.json
"""
import json
from argparse import ArgumentParser

from rdkit import Chem
from tqdm import tqdm

from megalodon.metrics.aimnet2.constrained_opt import parse_atom_contacts, resolve_contact_pairs


def iter_records(path):
    """Yield each SDF record's text."""
    record = []
    with open(path) as f:
        for line in f:
            if not record and not line.strip():
                continue
            record.append(line)
            if line.strip() == "$$$$":
                yield "".join(record)
                record = []


def main():
    parser = ArgumentParser()
    parser.add_argument("--sdf", type=str, required=True, help="SDF with ATOM_CONTACTS tags.")
    parser.add_argument("--output", type=str, required=True, help="Output .json path.")
    args = parser.parse_args()

    contacts = {}
    n_unreadable = n_unindexed = n_without = 0
    for block in tqdm(iter_records(args.sdf), desc="Reading"):
        mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=False)
        if mol is None:
            n_unreadable += 1
            continue
        title = block.splitlines()[0].strip()
        parsed, unindexed = parse_atom_contacts(block)
        n_unindexed += unindexed
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        pairs = resolve_contact_pairs(symbols, parsed)
        if title in contacts:
            print(f"WARNING: duplicate title {title!r}; keeping the first record.")
            continue
        contacts[title] = [list(p) for p in pairs]
        n_without += not pairs

    with open(args.output, "w") as f:
        json.dump(contacts, f)
    print(f"Wrote contacts for {len(contacts)} structures to {args.output} "
          f"({n_without} with no resolvable contacts, {n_unreadable} unreadable records, "
          f"{n_unindexed} contact lines without atom indices skipped).")


if __name__ == "__main__":
    main()
