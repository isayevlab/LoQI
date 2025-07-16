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
Process QM9 dataset and convert to PyTorch Geometric format.

This script processes QM9 molecular data and converts it to PyTorch Geometric
format while computing comprehensive molecular statistics.
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from utils_data import (
    mol_to_torch_geometric,
    compute_all_statistics,
    save_statistics,
    save_pyg_dataset,
    save_pickle,
    FULL_ATOM_ENCODER,
    DEFAULT_CHARGES_DICT
)


def read_qm9_sdf(sdf_path: Path) -> List[Chem.Mol]:
    """
    Read molecules from QM9 SDF file.
    
    Args:
        sdf_path: Path to SDF file
        
    Returns:
        List of RDKit molecule objects
    """
    print(f"Reading molecules from {sdf_path}")
    molecules = []
    
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
    
    for mol in tqdm(suppl, desc="Loading molecules"):
        if mol is not None:
            try:
                # Try to sanitize the molecule
                Chem.SanitizeMol(mol)
                molecules.append(mol)
            except Exception as e:
                print(f"Warning: Failed to sanitize molecule: {e}")
                continue
    
    print(f"Successfully loaded {len(molecules)} molecules.")
    return molecules


def read_qm9_properties(properties_path: Path) -> pd.DataFrame:
    """
    Read QM9 molecular properties from CSV file.
    
    Args:
        properties_path: Path to properties CSV file
        
    Returns:
        DataFrame with molecular properties
    """
    if not properties_path.exists():
        print(f"Warning: Properties file {properties_path} not found.")
        return None
        
    print(f"Reading molecular properties from {properties_path}")
    
    # QM9 property columns (from the original dataset)
    columns = [
        'mol_id', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 
        'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv'
    ]
    
    try:
        df = pd.read_csv(properties_path, names=columns)
        print(f"Loaded properties for {len(df)} molecules.")
        return df
    except Exception as e:
        print(f"Error reading properties file: {e}")
        return None


def filter_valid_molecules(molecules: List[Chem.Mol], 
                          atom_encoder: Dict[str, int]) -> List[Chem.Mol]:
    """
    Filter molecules to only include those with atoms in the encoder.
    
    Args:
        molecules: List of RDKit molecule objects
        atom_encoder: Dictionary mapping atom symbols to indices
        
    Returns:
        List of filtered molecules
    """
    print("Filtering molecules by atom types...")
    valid_molecules = []
    
    for mol in tqdm(molecules, desc="Filtering molecules"):
        if mol is None:
            continue
            
        # Check if all atoms are in the encoder
        valid = True
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in atom_encoder:
                valid = False
                break
                
        if valid:
            valid_molecules.append(mol)
    
    print(f"Filtered to {len(valid_molecules)} valid molecules "
          f"(removed {len(molecules) - len(valid_molecules)}).")
    return valid_molecules


def convert_molecules_to_pyg(molecules: List[Chem.Mol], 
                           atom_encoder: Dict[str, int],
                           properties_df: Optional[pd.DataFrame] = None) -> List:
    """
    Convert RDKit molecules to PyTorch Geometric format.
    
    Args:
        molecules: List of RDKit molecule objects
        atom_encoder: Dictionary mapping atom symbols to indices
        properties_df: Optional DataFrame with molecular properties
        
    Returns:
        List of PyTorch Geometric Data objects
    """
    print("Converting molecules to PyTorch Geometric format...")
    pyg_molecules = []
    
    for i, mol in enumerate(tqdm(molecules, desc="Converting molecules")):
        if mol is None:
            continue
            
        try:
            # Generate SMILES
            smiles = Chem.MolToSmiles(mol)
            
            # Convert to PyG format
            pyg_data = mol_to_torch_geometric(mol, smiles, atom_encoder)
            
            # Add molecular properties if available
            if properties_df is not None and i < len(properties_df):
                prop_row = properties_df.iloc[i]
                for col in properties_df.columns:
                    if col != 'mol_id':
                        setattr(pyg_data, col, float(prop_row[col]))
            
            # Add additional molecular descriptors
            pyg_data.molecular_weight = Descriptors.MolWt(mol)
            pyg_data.num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            pyg_data.tpsa = Descriptors.TPSA(mol)
            
            pyg_molecules.append(pyg_data)
            
        except Exception as e:
            print(f"Warning: Failed to convert molecule {i}: {e}")
            continue
    
    print(f"Successfully converted {len(pyg_molecules)} molecules.")
    return pyg_molecules


def split_qm9_dataset(pyg_molecules: List, 
                     random_seed: int = 42) -> tuple:
    """
    Split QM9 dataset into train/validation/test sets using the standard QM9 split.
    
    Args:
        pyg_molecules: List of PyTorch Geometric Data objects
        random_seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    print(f"Splitting dataset using standard QM9 split...")
    
    np.random.seed(random_seed)
    n_total = len(pyg_molecules)
    
    # Standard QM9 split: 100k train, 10% test, remainder validation
    n_train = 100000
    n_test = int(0.1 * n_total)
    n_val = n_total - (n_train + n_test)
    
    if n_total < n_train:
        print(f"Warning: Dataset size ({n_total}) is smaller than standard training size (100k)")
        print("Using 80% for training instead...")
        n_train = int(0.8 * n_total)
        n_test = int(0.1 * n_total)
        n_val = n_total - (n_train + n_test)
    
    # Shuffle dataset indices
    indices = np.random.permutation(n_total)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_data = [pyg_molecules[i] for i in train_indices]
    val_data = [pyg_molecules[i] for i in val_indices]
    test_data = [pyg_molecules[i] for i in test_indices]
    
    print(f"Split sizes - Train: {len(train_data)}, "
          f"Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def process_qm9_split(split_data: List,
                     split_name: str,
                     save_processed_path: Path,
                     atom_encoder: Dict[str, int],
                     charges_dict: Dict[int, int]) -> None:
    """
    Process and save a single QM9 data split.
    
    Args:
        split_data: List of PyTorch Geometric Data objects
        split_name: Name of the split (train/val/test)
        save_processed_path: Path to save processed data
        atom_encoder: Dictionary mapping atom symbols to indices
        charges_dict: Dictionary mapping formal charges to indices
    """
    print(f"\nProcessing {split_name} split with {len(split_data)} molecules...")
    
    # Save PyTorch Geometric dataset
    dataset_path = save_processed_path / f"{split_name}_h.pt"
    save_pyg_dataset(split_data, dataset_path)
    print(f"Saved PyG dataset to {dataset_path}")
    
    # Compute and save statistics
    print(f"Computing statistics for {split_name} split...")
    statistics = compute_all_statistics(
        split_data,
        atom_encoder,
        charges_dict,
        additional_features=True
    )
    
    save_statistics(statistics, save_processed_path, split_name)
    print(f"Saved statistics for {split_name} split.")


def main(args: argparse.Namespace) -> None:
    """
    Main processing function for QM9 dataset.
    
    Args:
        args: Command line arguments
    """
    print("QM9 Dataset Processing")
    print("=" * 50)
    
    # Setup paths
    qm9_sdf_path = Path(args.qm9_sdf_path)
    if not qm9_sdf_path.exists():
        raise FileNotFoundError(f"QM9 SDF file not found: {qm9_sdf_path}")
    
    save_data_path = Path(args.save_data_folder)
    save_data_path.mkdir(parents=True, exist_ok=True)
    save_processed_path = save_data_path / "processed"
    save_processed_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Input SDF: {qm9_sdf_path}")
    print(f"Output directory: {save_processed_path}")
    
    # Load molecular properties if provided
    properties_df = None
    if args.properties_csv:
        properties_path = Path(args.properties_csv)
        properties_df = read_qm9_properties(properties_path)
    
    # Read molecules from SDF
    molecules = read_qm9_sdf(qm9_sdf_path)
    
    # Filter molecules by atom types
    valid_molecules = filter_valid_molecules(molecules, FULL_ATOM_ENCODER)
    
    # Convert to PyTorch Geometric format
    pyg_molecules = convert_molecules_to_pyg(
        valid_molecules, 
        FULL_ATOM_ENCODER,
        properties_df
    )
    
    if len(pyg_molecules) == 0:
        print("Error: No valid molecules found after processing.")
        return
    
    # Split dataset
    train_data, val_data, test_data = split_qm9_dataset(
        pyg_molecules,
        random_seed=args.random_seed
    )
    
    # Process and save each split
    for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        if len(split_data) > 0:
            try:
                process_qm9_split(
                    split_data, split_name, save_processed_path,
                    FULL_ATOM_ENCODER, DEFAULT_CHARGES_DICT
                )
            except Exception as e:
                print(f"Error processing {split_name} split: {e}")
                continue
    
    # Save split information
    split_info = {
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'total_size': len(pyg_molecules),
        'atom_encoder': FULL_ATOM_ENCODER,
        'charges_dict': DEFAULT_CHARGES_DICT
    }
    save_pickle(split_info, save_processed_path / "split_info.pickle")
    
    print("\n" + "=" * 50)
    print("QM9 dataset processing completed successfully!")
    print(f"Total molecules processed: {len(pyg_molecules)}")
    print(f"All processed data saved to: {save_processed_path}")


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Process QM9 dataset and convert to PyTorch Geometric format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--qm9_sdf_path",
        type=str,
        required=True,
        help="Path to QM9 SDF file containing molecular structures"
    )
    
    parser.add_argument(
        "--save_data_folder",
        type=str,
        required=True,
        help="Path to directory where processed datasets will be saved"
    )
    
    parser.add_argument(
        "--properties_csv",
        type=str,
        default=None,
        help="Optional path to CSV file containing molecular properties"
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible dataset splits"
    )
    
    return parser


if __name__ == "__main__":
    parser = setup_argument_parser()
    args = parser.parse_args()
    main(args)
