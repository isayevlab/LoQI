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
Data processing utilities for molecular datasets.

This module contains common functionality for processing molecular data,
including geometry calculations, PyTorch Geometric conversions, and statistics computation.
"""

import pickle
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from tqdm import tqdm


# Constants
X_MAP = {
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
    "hybridization": [
        Chem.rdchem.HybridizationType.UNSPECIFIED,
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP2D,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.OTHER,
    ],
}

FULL_ATOM_ENCODER = {
    "H": 0, "B": 1, "C": 2, "N": 3, "O": 4, "F": 5,
    "Al": 6, "Si": 7, "P": 8, "S": 9, "Cl": 10, "As": 11,
    "Br": 12, "I": 13, "Hg": 14, "Bi": 15, "Se": 16
}

DEFAULT_CHARGES_DICT = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5}


# ===============================
# Geometry Calculation Functions
# ===============================

def generate_canonical_key(*components: Any) -> Tuple:
    """
    Generate a canonical key for any molecular component (atoms, bonds).
    This works for angles, bond lengths, and torsions.
    
    Args:
        *components: Variable number of components to create key from
        
    Returns:
        Canonical tuple key (min of forward and reverse order)
    """
    key1 = tuple(components)
    key2 = tuple(reversed(components))
    return min(key1, key2)


def compute_bond_lengths(rdkit_mol: Chem.Mol) -> Dict[Tuple, List]:
    """
    Compute bond lengths for all bonds in a molecule.
    
    Args:
        rdkit_mol: RDKit molecule object with conformer
        
    Returns:
        Dictionary mapping canonical bond keys to [lengths_list, count]
    """
    bond_lengths = {}
    conf = rdkit_mol.GetConformer()

    for bond in rdkit_mol.GetBonds():
        idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom1_type = rdkit_mol.GetAtomWithIdx(idx1).GetAtomicNum()
        atom2_type = rdkit_mol.GetAtomWithIdx(idx2).GetAtomicNum()
        bond_type_numeric = int(bond.GetBondType())
        
        length = rdMolTransforms.GetBondLength(conf, idx1, idx2)
        key = generate_canonical_key(atom1_type, bond_type_numeric, atom2_type)
        
        if key not in bond_lengths:
            bond_lengths[key] = [[], 0]
        bond_lengths[key][0].append(length)
        bond_lengths[key][1] += 1
        
    return bond_lengths


def compute_bond_angles(rdkit_mol: Chem.Mol) -> Dict[Tuple, List]:
    """
    Compute bond angles for all angle triplets in a molecule.
    
    Args:
        rdkit_mol: RDKit molecule object with conformer
        
    Returns:
        Dictionary mapping canonical angle keys to [angles_list, count]
    """
    bond_angles = {}
    conf = rdkit_mol.GetConformer()

    for atom in rdkit_mol.GetAtoms():
        neighbors = atom.GetNeighbors()
        if len(neighbors) < 2:
            continue

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                idx1, idx2, idx3 = neighbors[i].GetIdx(), atom.GetIdx(), neighbors[j].GetIdx()
                
                atom1_type = rdkit_mol.GetAtomWithIdx(idx1).GetAtomicNum()
                atom2_type = rdkit_mol.GetAtomWithIdx(idx2).GetAtomicNum()
                atom3_type = rdkit_mol.GetAtomWithIdx(idx3).GetAtomicNum()
                
                bond_type_1 = int(rdkit_mol.GetBondBetweenAtoms(idx1, idx2).GetBondType())
                bond_type_2 = int(rdkit_mol.GetBondBetweenAtoms(idx2, idx3).GetBondType())

                angle = rdMolTransforms.GetAngleDeg(conf, idx1, idx2, idx3)
                key = generate_canonical_key(atom1_type, bond_type_1, atom2_type, bond_type_2, atom3_type)
                
                if key not in bond_angles:
                    bond_angles[key] = [[], 0]
                bond_angles[key][0].append(angle)
                bond_angles[key][1] += 1

    return bond_angles


def compute_torsion_angles(rdkit_mol: Chem.Mol) -> Dict[Tuple, List]:
    """
    Compute torsion angles for all torsion quadruplets in a molecule.
    
    Args:
        rdkit_mol: RDKit molecule object with conformer
        
    Returns:
        Dictionary mapping canonical torsion keys to [angles_list, count]
    """
    torsion_smarts = "[!$(*#*)&!D1]~[!$(*#*)&!D1]"
    torsion_query = Chem.MolFromSmarts(torsion_smarts)
    torsion_angles = {}
    conf = rdkit_mol.GetConformer()

    torsion_matches = rdkit_mol.GetSubstructMatches(torsion_query)

    for match in torsion_matches:
        idx2, idx3 = match[0], match[1]
        bond = rdkit_mol.GetBondBetweenAtoms(idx2, idx3)

        for b1 in rdkit_mol.GetAtomWithIdx(idx2).GetBonds():
            if b1.GetIdx() == bond.GetIdx():
                continue
            idx1 = b1.GetOtherAtomIdx(idx2)
            
            for b2 in rdkit_mol.GetAtomWithIdx(idx3).GetBonds():
                if b2.GetIdx() == bond.GetIdx() or b2.GetIdx() == b1.GetIdx():
                    continue
                idx4 = b2.GetOtherAtomIdx(idx3)
                if idx4 == idx1:
                    continue

                atom1_type = rdkit_mol.GetAtomWithIdx(idx1).GetAtomicNum()
                atom2_type = rdkit_mol.GetAtomWithIdx(idx2).GetAtomicNum()
                atom3_type = rdkit_mol.GetAtomWithIdx(idx3).GetAtomicNum()
                atom4_type = rdkit_mol.GetAtomWithIdx(idx4).GetAtomicNum()
                
                bond_type_1 = int(b1.GetBondType())
                bond_type_2 = int(bond.GetBondType())
                bond_type_3 = int(b2.GetBondType())

                angle = rdMolTransforms.GetDihedralDeg(conf, idx1, idx2, idx3, idx4)
                key = generate_canonical_key(
                    atom1_type, bond_type_1, atom2_type, bond_type_2,
                    atom3_type, bond_type_3, atom4_type
                )
                
                if key not in torsion_angles:
                    torsion_angles[key] = [[], 0]
                torsion_angles[key][0].append(angle)
                torsion_angles[key][1] += 1

    return torsion_angles


def collect_geometry_parallel(molecules: List[Chem.Mol], 
                            compute_function: callable, 
                            num_processes: int = 4) -> Dict:
    """
    Parallelize geometry computation using a specific function.
    
    Args:
        molecules: List of RDKit molecule objects
        compute_function: Function to compute geometry (bond_lengths, angles, etc.)
        num_processes: Number of processes for parallelization (1 for sequential)
        
    Returns:
        Dictionary with aggregated geometry statistics
    """
    diff_sums = defaultdict(lambda: [[], 0])

    if num_processes == 1:
        results = []
        for mol in tqdm(molecules, desc="Processing molecules sequentially"):
            result = compute_function(mol)
            results.append(result)
    else:
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(compute_function, molecules),
                total=len(molecules),
                desc="Processing molecules in parallel"
            ))

    # Aggregate results
    for result in results:
        for key, (values_list, count) in result.items():
            diff_sums[key][0].extend(values_list)
            diff_sums[key][1] += count
            
    return dict(diff_sums)


def collect_all_geometries(data_list: List[Data]) -> Tuple[Dict, Dict, Dict]:
    """
    Collect bond lengths, angles, and torsions for a dataset.
    
    Args:
        data_list: List of PyTorch Geometric Data objects with 'mol' attribute
        
    Returns:
        Tuple of (bond_lengths, bond_angles, torsion_angles) dictionaries
    """
    molecules = [data.mol for data in data_list]
    
    bond_lengths = collect_geometry_parallel(molecules, compute_bond_lengths, num_processes=1)
    bond_angles = collect_geometry_parallel(molecules, compute_bond_angles, num_processes=1)
    torsion_angles = collect_geometry_parallel(molecules, compute_torsion_angles, num_processes=1)
    
    return bond_lengths, bond_angles, torsion_angles


# ===============================
# PyTorch Geometric Conversion
# ===============================

def mol_to_torch_geometric(mol: Chem.Mol, 
                          smiles: str, 
                          atom_encoder: Optional[Dict] = None) -> Data:
    """
    Convert RDKit molecule to PyTorch Geometric Data object.
    
    Args:
        mol: RDKit molecule object with conformer
        smiles: SMILES string
        atom_encoder: Dictionary mapping atom symbols to indices
        
    Returns:
        PyTorch Geometric Data object
    """
    if atom_encoder is None:
        atom_encoder = FULL_ATOM_ENCODER
        
    # Build adjacency matrix and edge information
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4  # Aromatic bonds
    edge_attr = bond_types.to(torch.uint8)

    # Get 3D coordinates
    pos = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    
    # Extract atom features
    atom_types, all_charges, is_aromatic, is_in_ring, sp_hybridization = [], [], [], [], []

    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
        all_charges.append(atom.GetFormalCharge())
        is_aromatic.append(X_MAP["is_aromatic"].index(atom.GetIsAromatic()))
        is_in_ring.append(X_MAP["is_in_ring"].index(atom.IsInRing()))
        sp_hybridization.append(X_MAP["hybridization"].index(atom.GetHybridization()))

    # Convert to tensors
    atom_types = torch.tensor(atom_types, dtype=torch.uint8)
    all_charges = torch.tensor(all_charges, dtype=torch.int8)
    is_aromatic = torch.tensor(is_aromatic, dtype=torch.uint8)
    is_in_ring = torch.tensor(is_in_ring, dtype=torch.uint8)
    hybridization = torch.tensor(sp_hybridization, dtype=torch.uint8)

    return Data(
        x=atom_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        charges=all_charges,
        smiles=smiles,
        is_aromatic=is_aromatic,
        is_in_ring=is_in_ring,
        hybridization=hybridization,
        mol=mol
    )


def raw_to_pyg(rdkit_mol: Chem.Mol, 
               coords: np.ndarray, 
               atom_encoder: Optional[Dict] = None) -> Data:
    """
    Convert raw RDKit molecule and coordinates to PyTorch Geometric format.
    
    Args:
        rdkit_mol: RDKit molecule object (topology only)
        coords: 3D coordinates array of shape (n_atoms, 3)
        atom_encoder: Dictionary mapping atom symbols to indices
        
    Returns:
        PyTorch Geometric Data object
        
    Raises:
        ValueError: If coordinates shape doesn't match number of atoms
    """
    # Create a copy and remove existing conformers
    mol = Chem.Mol(rdkit_mol)
    mol.RemoveAllConformers()
    
    # Validate coordinates
    coords = np.asarray(coords)
    if coords.shape != (mol.GetNumAtoms(), 3):
        raise ValueError(
            f"Coordinates shape {coords.shape} doesn't match expected "
            f"({mol.GetNumAtoms()}, 3)"
        )
    
    # Add new conformer with provided coordinates
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, coords[i].astype(float))
    mol.AddConformer(conf)
    
    # Generate SMILES and convert to PyG
    smiles = Chem.MolToSmiles(mol)
    return mol_to_torch_geometric(mol, smiles, atom_encoder)


# ===============================
# Statistics Computation
# ===============================

def compute_node_counts(data_list: List[Data]) -> Counter:
    """Compute distribution of number of nodes per molecule."""
    print("Computing node counts...")
    node_counts = Counter()
    for data in data_list:
        node_counts[data.num_nodes] += 1
    print("Done.")
    return node_counts


def compute_atom_type_counts(data_list: List[Data], num_classes: int) -> np.ndarray:
    """Compute normalized distribution of atom types."""
    print("Computing atom type distribution...")
    counts = np.zeros(num_classes)
    for data in data_list:
        one_hot = torch.nn.functional.one_hot(data.x.long(), num_classes=num_classes)
        counts += one_hot.sum(dim=0).numpy()
    counts = counts / counts.sum()
    print("Done.")
    return counts


def compute_edge_counts(data_list: List[Data], num_bond_types: int = 5) -> np.ndarray:
    """Compute normalized distribution of edge types including non-edges."""
    print("Computing edge counts...")
    counts = np.zeros(num_bond_types)

    for data in data_list:
        total_pairs = data.num_nodes * (data.num_nodes - 1)
        num_edges = data.edge_attr.shape[0]
        num_non_edges = total_pairs - num_edges
        assert num_non_edges >= 0

        edge_types = torch.nn.functional.one_hot(
            data.edge_attr.long() - 1, 
            num_classes=num_bond_types - 1
        ).sum(dim=0).numpy()
        
        counts[0] += num_non_edges  # Non-edges
        counts[1:] += edge_types    # Actual edges

    counts = counts / counts.sum()
    print("Done.")
    return counts


def compute_charge_counts(data_list: List[Data], 
                         num_classes: int, 
                         charges_dict: Dict[int, int]) -> np.ndarray:
    """Compute normalized distribution of formal charges per atom type."""
    print("Computing charge counts...")
    counts = np.zeros((num_classes, len(charges_dict)))

    for data in data_list:
        for atom_type, charge in zip(data.x, data.charges):
            assert charge.item() in charges_dict
            counts[atom_type.item(), charges_dict[charge.item()]] += 1

    # Normalize per atom type
    row_sums = np.sum(counts, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    counts = counts / row_sums
    print("Done.")
    return counts


def compute_valency_counts(data_list: List[Data], 
                          atom_encoder: Dict[str, int]) -> Dict[str, Counter]:
    """Compute normalized valency distribution per atom type."""
    atom_decoder = {v: k for k, v in atom_encoder.items()}
    print("Computing valency counts...")
    valencies = {atom_type: Counter() for atom_type in atom_encoder.keys()}

    for data in data_list:
        edge_attr = data.edge_attr.float()
        edge_attr[edge_attr == 4] = 1.5  # Convert aromatic bonds back to 1.5

        for atom_idx in range(data.num_nodes):
            # Find edges connected to this atom
            connected_edges = edge_attr[data.edge_index[0] == atom_idx]
            valency = connected_edges.sum().item()
            atom_type = atom_decoder[data.x[atom_idx].item()]
            valencies[atom_type][valency] += 1

    # Normalize
    for atom_type in valencies.keys():
        total = sum(valencies[atom_type].values())
        if total > 0:
            for valency in valencies[atom_type]:
                valencies[atom_type][valency] /= total
    print("Done.")
    return valencies


def compute_additional_feature_counts(data_list: List[Data], 
                                    features: List[str] = None) -> Dict[str, np.ndarray]:
    """Compute normalized distributions for additional molecular features."""
    if features is None:
        features = ["is_aromatic", "is_in_ring", "hybridization"]
        
    print(f"Computing counts for features: {features}")
    num_classes_list = [len(X_MAP[feature]) for feature in features]
    counts_list = [np.zeros(num_classes) for num_classes in num_classes_list]

    for data in data_list:
        for i, (feature, num_classes) in enumerate(zip(features, num_classes_list)):
            one_hot = torch.nn.functional.one_hot(
                getattr(data, feature).long(), 
                num_classes=num_classes
            )
            counts_list[i] += one_hot.sum(dim=0).numpy()

    # Normalize
    for i in range(len(counts_list)):
        counts_list[i] /= counts_list[i].sum()
    print("Done.")
    
    return {feature: counts for feature, counts in zip(features, counts_list)}


def compute_all_statistics(data_list: List[Data], 
                          atom_encoder: Dict[str, int],
                          charges_dict: Optional[Dict[int, int]] = None,
                          additional_features: bool = True) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a molecular dataset.
    
    Args:
        data_list: List of PyTorch Geometric Data objects
        atom_encoder: Dictionary mapping atom symbols to indices
        charges_dict: Dictionary mapping formal charges to indices
        additional_features: Whether to compute additional molecular features
        
    Returns:
        Dictionary containing all computed statistics
    """
    if charges_dict is None:
        charges_dict = DEFAULT_CHARGES_DICT
        
    print("Computing comprehensive molecular statistics...")
    
    # Basic statistics
    num_nodes = compute_node_counts(data_list)
    atom_types = compute_atom_type_counts(data_list, len(atom_encoder))
    bond_types = compute_edge_counts(data_list)
    charge_types = compute_charge_counts(data_list, len(atom_encoder), charges_dict)
    charge_prior = (charge_types * atom_types.reshape(-1, 1)).sum(0)
    valencies = compute_valency_counts(data_list, atom_encoder)
    
    # Geometry statistics
    bond_lengths, bond_angles, torsion_angles = collect_all_geometries(data_list)
    
    # Additional features
    additional_stats = None
    if additional_features:
        additional_stats = compute_additional_feature_counts(data_list)
    
    statistics = {
        'num_nodes': num_nodes,
        'atom_types': atom_types,
        'bond_types': bond_types,
        'charge_types': charge_types,
        'charge_prior': charge_prior,
        'valencies': valencies,
        'bond_lengths': bond_lengths,
        'bond_angles': bond_angles,
        'torsion_angles': torsion_angles,
        'smiles': [data.smiles for data in data_list]
    }
    
    if additional_stats:
        statistics['additional_features'] = additional_stats
        
    print("Statistics computation completed.")
    return statistics


# ===============================
# File I/O Utilities
# ===============================

def save_pickle(data: Any, path: Union[str, Path]) -> None:
    """Save data to pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load data from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pyg_dataset(pyg_batch: List[Data], save_path: Union[str, Path]) -> None:
    """Save PyTorch Geometric dataset using collate."""
    collated_data = collate(
        pyg_batch[0].__class__, 
        pyg_batch,
        increment=False,
        add_batch=False
    )
    torch.save(collated_data[:2], save_path)


def save_statistics(statistics: Dict[str, Any], 
                   save_path: Path, 
                   split_name: str) -> None:
    """
    Save computed statistics to files.
    
    Args:
        statistics: Dictionary of computed statistics
        save_path: Directory path to save files
        split_name: Name of the data split (train/val/test)
    """
    save_pickle(statistics['num_nodes'], save_path / f"{split_name}_n_h.pickle")
    np.save(save_path / f"{split_name}_atom_types_h.npy", statistics['atom_types'])
    np.save(save_path / f"{split_name}_bond_types_h.npy", statistics['bond_types'])
    np.save(save_path / f"{split_name}_charges_h.npy", statistics['charge_types'])
    np.save(save_path / f"{split_name}_charges_prior_h.npy", statistics['charge_prior'])
    save_pickle(statistics['valencies'], save_path / f"{split_name}_valency_h.pickle")
    save_pickle(statistics['smiles'], save_path / f"{split_name}_smiles.pickle")
    save_pickle(statistics['bond_lengths'], save_path / f"{split_name}_bond_lengths_h.pickle")
    save_pickle(statistics['bond_angles'], save_path / f"{split_name}_angles_h.pickle")
    save_pickle(statistics['torsion_angles'], save_path / f"{split_name}_dihedrals_h.pickle")
    
    if 'additional_features' in statistics:
        for feature_name, values in statistics['additional_features'].items():
            np.save(save_path / f"{split_name}_{feature_name}_h.npy", values)


def read_pickle_file(filepath: Path) -> Any:
    """Read data from pickle file."""
    return pickle.loads(filepath.read_bytes()) 