# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Process GEOM dataset and convert to PyTorch Geometric format.

This script processes raw GEOM molecular data and converts it to PyTorch Geometric
format while computing comprehensive molecular statistics.
"""

import argparse
from pathlib import Path
from typing import List, Tuple


from tqdm import tqdm

from utils_data import (
    FULL_ATOM_ENCODER,
    DEFAULT_CHARGES_DICT,
    mol_to_torch_geometric,
    raw_to_pyg,
    compute_all_statistics,
    save_statistics,
    save_pyg_dataset,
    load_pickle,
)


def process_dataset(raw_dataset: dict, sdf_topologies: dict) -> List:
    """
    Process raw dataset by matching molecular IDs with topologies and coordinates.
    
    Args:
        raw_dataset: Dictionary containing molecular IDs and coordinates
        sdf_topologies: Dictionary mapping molecule IDs to RDKit molecule objects
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    molecules = []
    
    print(f"Processing {len(sdf_topologies)} molecules...")
    for molecule_id, rdkit_mol in tqdm(sdf_topologies.items(), desc="Converting molecules"):
        # Find matching coordinates for this molecule ID
        mol_id_bytes = bytes(molecule_id + " ", 'utf-8')
        mask = (mol_id_bytes == raw_dataset["mol_id"]).astype(bool)
        
        if mask.sum() > 0:
            coords = raw_dataset["coord"][mask][0]
            try:
                mol_data = raw_to_pyg(rdkit_mol, coords)
                molecules.append(mol_data)
            except Exception as e:
                print(f"Warning: Failed to process molecule {molecule_id}: {e}")
                continue
    
    print(f"Successfully processed {len(molecules)} molecules.")
    return molecules


def process_geom_split(data: List[Tuple], 
                      split_name: str, 
                      save_processed_path: Path,
                      n_conformers: int = 5,
                      limit_molecules: int = None) -> None:
    """
    Process a single data split (train/val/test) from GEOM dataset.
    
    Args:
        data: List of tuples containing (smiles, conformers)
        split_name: Name of the split (train/val/test)
        save_processed_path: Path to save processed data
        n_conformers: Maximum number of conformers to use per molecule
        limit_molecules: Maximum number of molecules to process (for testing)
    """
    print(f"\nProcessing {split_name} split...")
    
    # Limit molecules for testing if specified
    if limit_molecules is not None:
        data = data[:limit_molecules]
        print(f"Limited to {len(data)} molecules for testing.")
    
    pyg_molecules = []
    
    for smiles, conformers in tqdm(data, desc=f"Processing {split_name} molecules"):
        # Limit number of conformers
        conformers = conformers[:n_conformers]
        
        for conformer in conformers:
            try:
                pyg_data = mol_to_torch_geometric(conformer, smiles)
                pyg_molecules.append(pyg_data)
            except Exception as e:
                print(f"Warning: Failed to process conformer for {smiles}: {e}")
                continue
    
    print(f"Converted {len(pyg_molecules)} conformers for {split_name} split.")
    
    # Save PyTorch Geometric dataset
    dataset_path = save_processed_path / f"{split_name}_h.pt"
    save_pyg_dataset(pyg_molecules, dataset_path)
    print(f"Saved PyG dataset to {dataset_path}")
    
    # Compute and save statistics
    print(f"Computing statistics for {split_name} split...")
    statistics = compute_all_statistics(
        pyg_molecules,
        FULL_ATOM_ENCODER,
        DEFAULT_CHARGES_DICT,
        additional_features=True
    )
    
    save_statistics(statistics, save_processed_path, split_name)
    print(f"Saved statistics for {split_name} split.")


def load_geom_data(raw_data_dir: Path) -> Tuple[List, List, List]:
    """
    Load GEOM dataset splits from pickle files.
    
    Args:
        raw_data_dir: Path to directory containing raw data files
        
    Returns:
        Tuple of (validation, train, test) data lists
        
    Raises:
        FileNotFoundError: If required data files are not found
    """
    data_files = {
        'val': raw_data_dir / "val_data.pickle",
        'train': raw_data_dir / "train_data.pickle", 
        'test': raw_data_dir / "test_data.pickle"
    }
    
    # Check if all files exist
    for split, filepath in data_files.items():
        if not filepath.exists():
            raise FileNotFoundError(f"Required data file not found: {filepath}")
    
    print("Loading GEOM dataset splits...")
    splits_data = {}
    for split, filepath in data_files.items():
        print(f"Loading {split} data from {filepath}")
        splits_data[split] = load_pickle(filepath)
        print(f"Loaded {len(splits_data[split])} molecules for {split} split.")
    
    return splits_data['val'], splits_data['train'], splits_data['test']


def create_output_directories(save_data_folder: Path) -> Path:
    """
    Create necessary output directories.
    
    Args:
        save_data_folder: Base path for saving data
        
    Returns:
        Path to processed data directory
    """
    save_data_path = Path(save_data_folder)
    save_data_path.mkdir(parents=True, exist_ok=True)
    
    save_processed_path = save_data_path / "processed"
    save_processed_path.mkdir(parents=True, exist_ok=True)
    
    return save_processed_path


def main(args: argparse.Namespace) -> None:
    """
    Main processing function for GEOM dataset.
    
    Args:
        args: Command line arguments containing raw_data_dir and save_data_folder
    """
    print("GEOM Dataset Processing")
    print("=" * 50)
    
    # Setup paths
    raw_data_dir = Path(args.raw_data_dir)
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")
    
    save_processed_path = create_output_directories(args.save_data_folder)
    print(f"Output directory: {save_processed_path}")
    
    # Load data splits
    try:
        val_data, train_data, test_data = load_geom_data(raw_data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Set molecule limit for testing
    limit_molecules = 10 if args.test_mode else None
    if args.test_mode:
        print(f"\n🧪 TEST MODE: Processing only {limit_molecules} molecules per split")
    
    # Process each split
    n_conformers = 5
    print(f"\nUsing maximum {n_conformers} conformers per molecule.")
    
    for split_name, data in [("val", val_data), ("train", train_data), ("test", test_data)]:
        try:
            process_geom_split(data, split_name, save_processed_path, n_conformers, limit_molecules)
        except Exception as e:
            print(f"Error processing {split_name} split: {e}")
            continue
    
    print("\n" + "=" * 50)
    print("GEOM dataset processing completed successfully!")
    print(f"All processed data saved to: {save_processed_path}")


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Process GEOM dataset and convert to PyTorch Geometric format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--raw_data_dir", 
        type=str, 
        required=True,
        help="Path to directory containing raw GEOM data pickle files"
    )
    
    parser.add_argument(
        "--save_data_folder", 
        type=str, 
        required=True,
        help="Path to directory where processed datasets will be saved"
    )
    
    parser.add_argument(
        "--test_mode", 
        action="store_true",
        help="Process only 10 molecules per split for testing purposes"
    )
    
    return parser


if __name__ == "__main__":
    parser = setup_argument_parser()
    args = parser.parse_args()
    main(args)

