"""
Utility functions for LoQI conformer generation app.
"""

import sys
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch

# Add src to path for imports
sys.path.append('/home/fnikitin/LoQI/src')

from megalodon.metrics.conformer_evaluation_callback import (
    write_coords_to_mol, convert_coords_to_np
)
from megalodon.metrics.molecule_evaluation_callback import full_atom_encoder
from megalodon.metrics.aimnet2.check_topology import check_topology
from megalodon.metrics.preserved_stereo import get_stereochemistry_descriptor


def smiles_to_mol(smiles, add_hs=True, embed_3d=True):
    """
    Convert SMILES string to RDKit molecule with optional 3D embedding.
    
    Args:
        smiles (str): SMILES string
        add_hs (bool): Whether to add hydrogens
        embed_3d (bool): Whether to embed 3D coordinates
        
    Returns:
        Chem.Mol or None: RDKit molecule object
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    if add_hs:
        mol = Chem.AddHs(mol)
    
    if embed_3d:
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        except:
            return None
    
    return mol


def add_stereo_bonds(mol, chi_bonds, ez_bonds, edge_index=None, edge_attr=None, from_3D=True):
    """Add stereochemistry edges to the molecular graph."""
    result = []
    if from_3D and mol.GetNumConformers() > 0:
        Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    else:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if bond.GetBondType() == Chem.BondType.DOUBLE and stereo in ez_bonds:
            idx_3, idx_4 = bond.GetStereoAtoms()
            atom_1, atom_2 = bond.GetBeginAtom(), bond.GetEndAtom()
            idx_1, idx_2 = atom_1.GetIdx(), atom_2.GetIdx()

            idx_5 = [nbr.GetIdx() for nbr in atom_1.GetNeighbors() if nbr.GetIdx() not in {idx_2, idx_3}]
            idx_6 = [nbr.GetIdx() for nbr in atom_2.GetNeighbors() if nbr.GetIdx() not in {idx_1, idx_4}]

            inv_stereo = Chem.BondStereo.STEREOE if stereo == Chem.BondStereo.STEREOZ else Chem.BondStereo.STEREOZ
            result.extend([(idx_3, idx_4, ez_bonds[stereo]), (idx_4, idx_3, ez_bonds[stereo])])

            if idx_5:
                result.extend([(idx_5[0], idx_4, ez_bonds[inv_stereo]), (idx_4, idx_5[0], ez_bonds[inv_stereo])])
            if idx_6:
                result.extend([(idx_3, idx_6[0], ez_bonds[inv_stereo]), (idx_6[0], idx_3, ez_bonds[inv_stereo])])
            if idx_5 and idx_6:
                result.extend([(idx_5[0], idx_6[0], ez_bonds[stereo]), (idx_6[0], idx_5[0], ez_bonds[stereo])])

        if bond.GetBeginAtom().HasProp('_CIPCode'):
            idx = bond.GetBeginAtom().GetIdx()
            chirality = bond.GetBeginAtom().GetProp('_CIPCode')
            neighbors = bond.GetBeginAtom().GetNeighbors()
            if all(n.HasProp("_CIPRank") for n in neighbors):
                sorted_neighbors = sorted(neighbors, key=lambda x: int(x.GetProp("_CIPRank")), reverse=True)
                sorted_neighbors = [a.GetIdx() for a in sorted_neighbors]
                a, b, c = sorted_neighbors[:3] if chirality == "R" else sorted_neighbors[:3][::-1]
                d = sorted_neighbors[-1]
                result.extend([
                    (a, d, chi_bonds[0]), (b, d, chi_bonds[0]), (c, d, chi_bonds[0]),
                    (d, a, chi_bonds[0]), (d, b, chi_bonds[0]), (d, c, chi_bonds[0]),
                    (b, a, chi_bonds[1]), (c, b, chi_bonds[1]), (a, c, chi_bonds[1])
                ])

    if not result:
        return edge_index, edge_attr
    new_edge_index = torch.tensor([[i, j] for i, j, _ in result], dtype=torch.long).T
    new_edge_attr = torch.tensor([b for _, _, b in result], dtype=torch.uint8)

    if edge_index is None:
        return new_edge_index, new_edge_attr
    edge_index = torch.cat([edge_index, new_edge_index], dim=1)
    edge_attr = torch.cat([edge_attr, new_edge_attr])
    return edge_index, edge_attr


def mol_to_torch_geometric_simple(mol, smiles):
    """
    Convert RDKit molecule to PyTorch Geometric Data object with stereochemistry edges.
    
    Args:
        mol (Chem.Mol): RDKit molecule
        smiles (str): SMILES string
        
    Returns:
        Data: PyTorch Geometric Data object
    """
    # Sanitize molecule
    Chem.SanitizeMol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    
    # Get adjacency matrix and edge information
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4  # Aromatic bonds
    edge_attr = bond_types.to(torch.uint8)
    
    # Get 3D coordinates if available
    if mol.GetNumConformers() > 0:
        pos = torch.tensor(mol.GetConformer().GetPositions()).float()
    else:
        pos = torch.zeros((mol.GetNumAtoms(), 3)).float()
    
    # Get atom features
    atom_types = torch.tensor([full_atom_encoder[atom.GetSymbol()] for atom in mol.GetAtoms()], dtype=torch.uint8)
    charges = torch.tensor([atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=torch.int8)
    
    # Add stereochemistry edges (CRITICAL for LoQI model!)
    chi_bonds = [7, 8]  # R/S stereochemistry edge types
    ez_bonds = {Chem.BondStereo.STEREOE: 5, Chem.BondStereo.STEREOZ: 6}  # E/Z edge types
    edge_index, edge_attr = add_stereo_bonds(mol, chi_bonds, ez_bonds, edge_index, edge_attr, from_3D=True)
    
    return Data(
        x=atom_types,
        edge_index=edge_index,
        edge_attr=edge_attr.to(torch.uint8),
        pos=pos,
        charges=charges,
        smiles=smiles,
        mol=mol,
        chemblid=mol.GetProp("_Name") if mol.HasProp("_Name") else ""
    )


def generate_conformers_batch(smiles, model, cfg, n_confs=10):
    """
    Generate multiple conformers for a given SMILES using the LoQI model.
    
    Args:
        smiles (str): SMILES string
        model: Trained LoQI model
        cfg: Configuration object
        n_confs (int): Number of conformers to generate
        
    Returns:
        tuple: (list of generated molecules, list of reference molecules, error message or None)
    """
    try:
        # Create base molecule
        mol = smiles_to_mol(smiles, add_hs=True, embed_3d=True)
        if mol is None:
            return None, None, "Invalid SMILES string or failed to embed 3D coordinates"
        
        # Create data list for batch processing
        data_list = []
        reference_mols = []
        for _ in range(n_confs):
            data = mol_to_torch_geometric_simple(mol, smiles)
            data_list.append(data)
            reference_mols.append(Chem.Mol(mol))  # Copy of original molecule for reference
        
        # Create batch and move to device
        batch = Batch.from_data_list(data_list).to(model.device)
        
        # Generate conformers
        with torch.no_grad():
            sample = model.sample(batch=batch, timesteps=cfg.interpolant.timesteps, pre_format=True)
            coords_list = convert_coords_to_np(sample)
        
        # Create molecules with generated coordinates
        generated_mols = []
        for coords in coords_list:
            new_mol = write_coords_to_mol(mol, coords)
            generated_mols.append(new_mol)
        
        return generated_mols, reference_mols, None
        
    except Exception as e:
        return None, None, str(e)


def create_sdf_content(molecules, energies_kcal=None, min_energy=None):
    """
    Create SDF content from a list of molecules with optional energy information.
    
    Args:
        molecules (list): List of RDKit molecules
        energies_kcal (array): Array of energies in kcal/mol
        min_energy (float): Minimum energy for relative energy calculation
        
    Returns:
        str: SDF content string
    """
    sdf_content = ""
    
    for i, mol in enumerate(molecules):
        mol_copy = Chem.Mol(mol)
        
        # Add energy properties if available
        if energies_kcal is not None:
            mol_copy.SetProp("Energy_kcal_mol", f"{energies_kcal[i]:.4f}")
            if min_energy is not None:
                mol_copy.SetProp("Relative_Energy_kcal_mol", f"{energies_kcal[i] - min_energy:.4f}")
            mol_copy.SetProp("Conformer_ID", str(i + 1))
            if min_energy is not None:
                mol_copy.SetProp("Is_Lowest_Energy", str(energies_kcal[i] == min_energy))
        
        sdf_content += Chem.MolToMolBlock(mol_copy)
        sdf_content += "$$$$\n"
    
    return sdf_content


def safe_filename_from_smiles(smiles, suffix=""):
    """
    Create a safe filename from a SMILES string.
    
    Args:
        smiles (str): SMILES string
        suffix (str): Optional suffix to add
        
    Returns:
        str: Safe filename
    """
    # Replace problematic characters
    safe_name = smiles.replace('/', '_').replace('\\', '_').replace(':', '_')
    safe_name = safe_name.replace('*', '_').replace('?', '_').replace('"', '_')
    safe_name = safe_name.replace('<', '_').replace('>', '_').replace('|', '_')
    
    # Limit length
    if len(safe_name) > 50:
        safe_name = safe_name[:50]
    
    return f"{safe_name}{suffix}"


def get_energy_statistics(energies_kcal, topology_results=None, stereo_results=None):
    """
    Calculate energy statistics from an array of energies.
    
    Args:
        energies_kcal (array): Array of energies in kcal/mol
        topology_results (dict): Topology preservation results
        stereo_results (dict): Stereochemistry preservation results
        
    Returns:
        dict: Dictionary with energy statistics (relative to minimum)
    """
    min_energy = float(np.min(energies_kcal))
    min_idx = int(np.argmin(energies_kcal))
    
    # Find minimum among molecules with preserved topology and stereochemistry
    preserved_min_energy = None
    preserved_min_idx = None
    
    if topology_results and stereo_results:
        topology_preserved = topology_results.get('topology_results', [])
        stereo_preserved = stereo_results.get('stereo_results', {}).get('preserved_stereo', [])
        
        # If molecule has stereochemistry, require both topology and stereo preservation
        # If no stereochemistry, only require topology preservation
        has_stereo = stereo_results.get('has_stereochemistry', False)
        
        preserved_indices = []
        for i in range(len(energies_kcal)):
            topology_ok = i < len(topology_preserved) and topology_preserved[i]
            
            if has_stereo:
                stereo_ok = i < len(stereo_preserved) and stereo_preserved[i]
                if topology_ok and stereo_ok:
                    preserved_indices.append(i)
            else:
                if topology_ok:
                    preserved_indices.append(i)
        
        if preserved_indices:
            preserved_energies = [energies_kcal[i] for i in preserved_indices]
            preserved_min_energy = float(np.min(preserved_energies))
            preserved_min_idx = preserved_indices[np.argmin(preserved_energies)]
    
    return {
        "min_energy": min_energy,
        "max_relative_energy": float(np.max(energies_kcal) - min_energy),
        "mean_relative_energy": float(np.mean(energies_kcal) - min_energy),
        "energy_range": float(np.max(energies_kcal) - min_energy),
        "min_idx": min_idx,
        "preserved_min_energy": preserved_min_energy,
        "preserved_min_idx": preserved_min_idx,
        "has_preserved_conformers": preserved_min_idx is not None
    }


def check_topology_preservation(molecules):
    """
    Check topology preservation for a list of molecules.
    
    Args:
        molecules (list): List of RDKit molecules
        
    Returns:
        dict: Topology preservation statistics
    """
    try:
        topology_results = []
        for mol in molecules:
            if mol is None or mol.GetNumConformers() == 0:
                topology_results.append(False)
                continue
                
            adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
            coordinates = np.array(mol.GetConformer().GetPositions().tolist()).reshape(1, -1, 3)
            numbers = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
            
            result = check_topology(adjacency_matrix, numbers, coordinates)
            topology_results.append(bool(result[0]))
        
        preserved_count = sum(topology_results)
        total_count = len(topology_results)
        
        return {
            "topology_preserved_count": preserved_count,
            "topology_preserved_percentage": (preserved_count / total_count * 100) if total_count > 0 else 0.0,
            "topology_results": topology_results
        }
    except Exception as e:
        return {
            "topology_preserved_count": 0,
            "topology_preserved_percentage": 0.0,
            "topology_results": [False] * len(molecules),
            "error": str(e)
        }


def check_stereochemistry_preservation(generated_molecules, reference_molecules):
    """
    Check stereochemistry preservation between generated and reference molecules.
    Uses the same logic as StereoMetrics but processes molecules individually.
    
    Args:
        generated_molecules (list): List of generated RDKit molecules
        reference_molecules (list): List of reference RDKit molecules
        
    Returns:
        dict: Stereochemistry preservation statistics
    """
    if not reference_molecules or len(generated_molecules) != len(reference_molecules):
        return {
            "stereo_preserved_count": 0,
            "stereo_preserved_percentage": 0.0,
            "has_stereochemistry": False,
            "error": "Reference molecules not available or count mismatch"
        }
    
    preserved_stereo = []
    has_stereo = False
    
    # Process each molecule pair individually using StereoMetrics logic
    for mol, ref_mol in zip(generated_molecules, reference_molecules):
        if mol is None or ref_mol is None:
            preserved_stereo.append(False)
            continue
        
        # Make copies to avoid modifying originals
        mol_copy = Chem.Mol(mol)
        ref_copy = Chem.Mol(ref_mol)
        
        # Assign stereochemistry from 3D coordinates (same as StereoMetrics)
        Chem.rdmolops.AssignStereochemistryFrom3D(ref_copy)
        Chem.rdmolops.AssignStereochemistryFrom3D(mol_copy)
        
        # Get descriptors (same as StereoMetrics)
        sr, inv_sr, ez = get_stereochemistry_descriptor(mol_copy)
        ref_sr, _, ref_ez = get_stereochemistry_descriptor(ref_copy)
        
        # Check if molecule has stereochemistry
        if ref_sr or ref_ez:
            has_stereo = True
            
            # For distance-matrix based models, we should check both original and inverted molecule stereochemistry
            # and accept whichever gives better preservation (especially important for single stereocenters)
            
            # Option 1: Compare original molecule with reference
            rs_correct_orig = True
            if ref_sr:
                rs_correct_orig = (sr == ref_sr)
            ez_correct_orig = True
            if ref_ez:
                ez_correct_orig = (ez == ref_ez)
            preservation_orig = rs_correct_orig and ez_correct_orig
            
            # Option 2: Compare inverted molecule stereochemistry with reference
            # Invert the generated molecule's R/S descriptors (R↔S)
            sr_inv = inv_sr  # This is the inverted R/S descriptor from get_stereochemistry_descriptor
            rs_correct_inv = True
            if ref_sr:
                rs_correct_inv = (sr_inv == ref_sr)
            ez_correct_inv = True
            if ref_ez:
                ez_correct_inv = (ez == ref_ez)  # E/Z doesn't get inverted
            preservation_inv = rs_correct_inv and ez_correct_inv
            
            # Accept the better preservation (for single stereocenter, one should be 100%)
            preserved_stereo.append(preservation_orig or preservation_inv)
        else:
            # No stereochemistry to preserve
            preserved_stereo.append(True)
    
    # Calculate statistics
    preserved_count = sum(preserved_stereo)
    total_count = len(preserved_stereo)
    preserved_percentage = (preserved_count / total_count * 100) if total_count > 0 else 0.0
    
    return {
        "stereo_preserved_count": preserved_count,
        "stereo_preserved_percentage": preserved_percentage,
        "has_stereochemistry": has_stereo,
        "stereo_results": {
            "preserved_stereo": preserved_stereo
        }
    } 