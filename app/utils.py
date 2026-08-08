"""
Utility functions for LoQI conformer generation app.
"""

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Add src to path for imports
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))
sys.path.append(str(ROOT))

from megalodon.metrics.conformer_evaluation_callback import (
    write_coords_to_mol, convert_coords_to_np
)
from megalodon.data.adaptive_dataloader import AdaptiveBatchSampler
from megalodon.inference.validation import SUPPORTED_ELEMENTS, validate_smiles
from megalodon.metrics.molecule_evaluation_callback import full_atom_encoder
from megalodon.metrics.aimnet2.check_topology import check_topology
from megalodon.metrics.preserved_stereo import (
    get_stereochemistry_descriptor,
    prepare_mol_for_conformer_eval,
)


from data_processing.utils_data import add_stereo_bonds


def _clean_spurious_stereo(mol: Chem.Mol) -> Chem.Mol:
    """Keep only real tetrahedral carbon stereo while preserving E/Z stereo."""
    mol = Chem.RWMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    true_centers = {
        idx
        for idx, _ in Chem.FindMolChiralCenters(mol, includeUnassigned=False)
        if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6
    }
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in true_centers:
            atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return Chem.Mol(mol)


def render_molecule_svg(smiles: str, width: int = 900, height: int = 320) -> Optional[str]:
    """Render a molecule as a styled RDKit SVG similar to the RitS app."""
    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    mol = Chem.MolFromSmiles(smiles, ps)
    if mol is None:
        return None

    Chem.SanitizeMol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    mol = Chem.RemoveAllHs(mol)
    mol = _clean_spurious_stereo(mol)
    rdDepictor.Compute2DCoords(mol)
    num_atoms = mol.GetNumAtoms()

    bond_line_width = 3.0
    if num_atoms > 140:
        bond_line_width /= 4.0
    elif num_atoms > 70:
        bond_line_width /= 2.0

    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.bondLineWidth = bond_line_width
    opts.baseFontSize = 0.9
    opts.minFontSize = 14
    opts.maxFontSize = 32
    opts.padding = 0.05
    opts.addStereoAnnotation = True
    opts.clearBackground = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


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


def generate_conformers_batch(
        smiles,
        model,
        cfg,
        n_confs=10,
        generation_batch_size=None,
        atom_aware_batching=None,
        shuffle=None,
        target_molecule_size=None,
):
    """
    Generate multiple conformers for a given SMILES using the LoQI model.
    
    Args:
        smiles (str): SMILES string
        model: Trained LoQI model
        cfg: Configuration object
        n_confs (int): Number of conformers to generate
        
    Returns:
        tuple: (generated molecules, reference molecules, seconds per structure, error message)
    """
    def _validate_smiles_with_disconnected_fallback(input_smiles):
        mol_valid, _, validation_error = validate_smiles(input_smiles)
        if validation_error is None:
            return mol_valid, None
        if "Disconnected fragments are not supported" not in str(validation_error):
            return None, validation_error

        # Fallback validation path for disconnected systems (e.g., dimers).
        mol = Chem.MolFromSmiles(str(input_smiles).strip())
        if mol is None:
            return None, f"RDKit failed to parse SMILES: {input_smiles!r}."
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        mol_roundtrip = Chem.MolFromSmiles(canonical)
        if mol_roundtrip is None:
            return None, f"Revalidation failed after canonicalization: {canonical!r}."
        mol_h = Chem.AddHs(mol_roundtrip)
        for atom in mol_h.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in SUPPORTED_ELEMENTS:
                return None, (
                    f"Unsupported element: '{symbol}'. Supported: "
                    f"{', '.join(sorted(SUPPORTED_ELEMENTS))}."
                )
            if atom.GetNumRadicalElectrons() > 0:
                return None, (
                    f"Radical electrons are not supported "
                    f"(atom {atom.GetIdx()} {atom.GetSymbol()})."
                )
        return mol_h, (
            "Disconnected fragments detected. Proceeding in permissive mode "
            "(experimental for dimers/multimers)."
        )

    try:
        # Validate and revalidate input SMILES first.
        mol, validation_warning = _validate_smiles_with_disconnected_fallback(smiles)
        if validation_warning is not None:
            print(f"WARNING: {validation_warning}")
        if mol is None:
            return None, None, None, "SMILES validation failed."

        # Create data list for batch processing
        data_list = []
        reference_mols = []
        for _ in range(n_confs):
            data = mol_to_torch_geometric_simple(mol, smiles)
            data_list.append(data)
            reference_mols.append(Chem.Mol(mol))  # Copy of original molecule for reference

        if generation_batch_size is None:
            generation_batch_size = int(
                getattr(cfg.data, "inference_batch_size", getattr(cfg.data, "batch_size", n_confs))
            )
        generation_batch_size = max(1, int(generation_batch_size))
        if atom_aware_batching is None:
            atom_aware_batching = True
        if shuffle is None:
            shuffle = False
        if target_molecule_size is None:
            target_molecule_size = 50

        # Generate conformers in batches
        timesteps = getattr(cfg.interpolant, "timesteps", 25)
        t0 = time.perf_counter()
        coords_list = []
        if atom_aware_batching:
            sampler = AdaptiveBatchSampler(
                data_list,
                reference_batch_size=generation_batch_size,
                shuffle=shuffle,
                reference_size=target_molecule_size,
            )
            loader = DataLoader(data_list, batch_sampler=sampler)
        else:
            loader = DataLoader(data_list, batch_size=generation_batch_size, shuffle=shuffle)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(model.device)
                sample = model.sample(batch=batch, timesteps=timesteps, pre_format=True)
                coords_list.extend(convert_coords_to_np(sample))
        elapsed_s = time.perf_counter() - t0
        per_structure_s = elapsed_s / max(len(coords_list), 1)
        
        # Create molecules with generated coordinates
        generated_mols = []
        for coords in coords_list:
            new_mol = write_coords_to_mol(mol, coords)
            generated_mols.append(new_mol)
        
        return generated_mols, reference_mols, per_structure_s, None
        
    except Exception as e:
        return None, None, None, str(e)


def set_cfg_timesteps(cfg, timesteps):
    """Set timesteps for sampling without mutating caller-owned config unexpectedly."""
    cfg.interpolant.timesteps = int(timesteps)
    if "evaluation" in cfg and "timesteps" in cfg.evaluation:
        cfg.evaluation.timesteps = int(timesteps)
    return cfg


def select_unique_with_irmsd(molecules, rthr=0.125):
    """
    Select iRMSD-unique subset from molecules.

    Returns:
        tuple: (unique_molecules, selected_indices, error_message)
    """
    try:
        from irmsd import sorter_irmsd_rdkit  # type: ignore
    except Exception:
        return None, None, (
            "iRMSD is not installed. Install with: pip install irmsd"
        )

    try:
        # iinversion=2 disables inversion.
        groups, _ = sorter_irmsd_rdkit(
            molecules, rthr=float(rthr), iinversion=2, allcanon=True, printlvl=0
        )
        groups = np.asarray(groups).reshape(-1)
        if groups.shape[0] != len(molecules):
            return None, None, (
                f"iRMSD returned unexpected group shape {groups.shape}; expected ({len(molecules)},)."
            )

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
    except Exception as e:
        return None, None, f"iRMSD pruning failed: {e}"


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
    energies_kcal = np.asarray(energies_kcal, dtype=float)
    if energies_kcal.size == 0:
        return {
            "min_energy": None,
            "max_relative_energy": None,
            "mean_relative_energy": None,
            "energy_range": None,
            "min_idx": None,
            "preserved_min_energy": None,
            "preserved_min_idx": None,
            "has_preserved_conformers": False,
            "has_energies": False,
        }

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
        "has_preserved_conformers": preserved_min_idx is not None,
        "has_energies": True,
    }


def check_topology_preservation(molecules):
    """Check topology preservation for a list of molecules."""
    try:
        topology_results = []
        for mol in molecules:
            if mol is None or mol.GetNumConformers() == 0:
                topology_results.append(False)
                continue

            mol = prepare_mol_for_conformer_eval(mol)
            if mol is None:
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
            'topology_preserved_count': preserved_count,
            'topology_preserved_percentage': (preserved_count / total_count * 100) if total_count > 0 else 0.0,
            'topology_results': topology_results,
        }
    except Exception as e:
        return {
            'topology_preserved_count': 0,
            'topology_preserved_percentage': 0.0,
            'topology_results': [False] * len(molecules),
            'error': str(e),
        }


def check_stereochemistry_preservation(generated_molecules, reference_molecules):
    """Check stereochemistry preservation between generated and reference molecules."""
    if not reference_molecules or len(generated_molecules) != len(reference_molecules):
        return {
            'stereo_preserved_count': 0,
            'stereo_preserved_percentage': 0.0,
            'has_stereochemistry': False,
            'error': 'Reference molecules not available or count mismatch',
        }

    preserved_stereo = []
    has_stereo = False
    for mol, ref_mol in zip(generated_molecules, reference_molecules):
        if mol is None or ref_mol is None:
            preserved_stereo.append(False)
            continue

        mol_copy = prepare_mol_for_conformer_eval(mol, assign_from_3d=True)
        ref_copy = prepare_mol_for_conformer_eval(ref_mol, assign_from_3d=True)
        if mol_copy is None or ref_copy is None:
            preserved_stereo.append(False)
            continue

        sr, _, ez = get_stereochemistry_descriptor(mol_copy)
        ref_sr, _, ref_ez = get_stereochemistry_descriptor(ref_copy)

        if ref_sr or ref_ez:
            has_stereo = True
            rs_correct_orig = True if not ref_sr else (sr == ref_sr)
            ez_correct_orig = True if not ref_ez else (ez == ref_ez)
            preserved_stereo.append(rs_correct_orig and ez_correct_orig)
        else:
            preserved_stereo.append(True)

    preserved_count = sum(preserved_stereo)
    total_count = len(preserved_stereo)
    preserved_percentage = (preserved_count / total_count * 100) if total_count > 0 else 0.0
    return {
        'stereo_preserved_count': preserved_count,
        'stereo_preserved_percentage': preserved_percentage,
        'has_stereochemistry': has_stereo,
        'stereo_results': {'preserved_stereo': preserved_stereo},
    }
