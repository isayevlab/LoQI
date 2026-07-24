"""
Load a processed-dataset split directly via MoleculeDataset (the same way
sample_conformers_processed.py does) and dump it to an SDF file, using each molecule's SMILES
string as its name ("_Name" property).

Example:
    python scripts/convert_data_to_sdf.py \
        --input /data/jarret/loqi/csd_loqi_data/csd_loqi_monomers_dih-relax_DEFAULT-SPLIT/processed \
        --split test \
        --output /data/jarret/loqi/csd_loqi_data/csd_loqi_monomers_dih-relax_DEFAULT-SPLIT/test.sdf
"""
import os
from argparse import ArgumentParser

from rdkit import Chem
from rdkit.Chem import SDWriter
from tqdm import tqdm

from megalodon.data.molecule_dataset import MoleculeDataset


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Processed-dataset directory (e.g. .../processed).")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--output", type=str, required=True, help="Output .sdf path.")
    args = parser.parse_args()

    dataset_root = os.path.dirname(os.path.normpath(args.input))
    processed_folder = os.path.basename(os.path.normpath(args.input))
    dataset = MoleculeDataset(root=dataset_root, processed_folder=processed_folder, split=args.split)

    writer = SDWriter(args.output)
    n_written = 0
    for idx in tqdm(range(len(dataset)), desc="Writing"):
        data = dataset[idx]
        mol = Chem.Mol(data.mol)
        mol.SetProp("_Name", data.smiles)
        writer.write(mol)
        n_written += 1
    writer.close()
    print(f"Wrote {n_written} molecules from split '{args.split}' of '{args.input}' to {args.output}")


if __name__ == "__main__":
    main()
