import time
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
from rdkit import Chem
from torch import Tensor
from torch import nn

from megalodon.metrics.aimnet2.check_topology import check_topology
from megalodon.metrics.aimnet2.dsopt import group_opt
from megalodon.metrics.aimnet2.pair_geometry import (
    compute_bond_lengths_diff,
    compute_bond_angles_diff,
    compute_torsion_angles_diff,
)


def is_valid(mol, verbose=False):
    """
    Validate a molecule for single fragment and successful sanitization.

    Args:
        mol (Chem.Mol): RDKit molecule object.
        verbose (bool): Print error messages if validation fails.

    Returns:
        bool: True if valid, otherwise False.
    """
    if mol is None:
        return False

    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.KekulizeException as e:
        if verbose:
            print(f"Kekulization failed: {e}")
        return False
    except ValueError as e:
        if verbose:
            print(f"Sanitization failed: {e}")
        return False

    if len(Chem.GetMolFrags(mol)) > 1:
        if verbose:
            print("Molecule has multiple fragments.")
        return False

    return True


def collect_geometry(pairs, compute_function):
    """
    Compute geometry metrics for molecule pairs using a specified function.

    Args:
        pairs (list): List of (initial, optimized) RDKit molecule pairs.
        compute_function (callable): Function to compute geometry metrics.

    Returns:
        dict: Aggregated geometry metrics.
    """
    diff_sums = {}
    results = []

    for idx, pair in enumerate(pairs):
        # try:
            if is_valid(pair[0]) and is_valid(pair[1]):
                init = Chem.Mol(pair[0])
                Chem.SanitizeMol(init)
                opt = Chem.Mol(pair[1])
                Chem.SanitizeMol(opt)
                result = compute_function((init, opt))
                values = torch.cat([torch.tensor(v[0]) for v in result.values()])
                if torch.isnan(values.sum()):
                    print(f"Skipping molecule {idx} due to invalid result.")
                    continue
                results.append(result)
        # except Exception:
        #     continue  # Skip invalid or problematic pairs

    for result in results:
        for key, (diff_list, count) in result.items():
            if key not in diff_sums:
                diff_sums[key] = [[], 0]
            diff_sums[key][0].extend(diff_list)
            diff_sums[key][1] += count

    return diff_sums


def aggregate_dict(dct, agg_idx):
    """
    Aggregate dictionary metrics by a specific key index.

    Args:
        dct (dict): Dictionary of geometry metrics.
        agg_idx (int): Index to aggregate by.

    Returns:
        dict: Aggregated metrics.
    """
    res = {}
    for k, v in dct.items():
        key = k[agg_idx]
        if key not in res:
            res[key] = [[], 0]
        res[key][0].extend(v[0])
        res[key][1] += v[1]
    return res


def compute_distance(pairs, agg_idx, compute_function):
    """
    Compute weighted average distance for geometry metrics.

    Args:
        pairs (list): List of molecule pairs.
        agg_idx (int): Index for aggregation.
        compute_function (callable): Function to compute geometry differences.

    Returns:
        float: Weighted average distance.
    """
    result_dict = collect_geometry(pairs, compute_function)
    agg_res = aggregate_dict(result_dict, agg_idx)

    total_count = sum(v[1] for v in agg_res.values())
    weights = {k: v[1] / total_count for k, v in agg_res.items()}

    res_distance = sum(weights[k] * np.mean(agg_res[k][0]) for k in agg_res)
    return res_distance


def prepare_for_aimnet(rdkit_molecules, device="cpu"):
    """
    Prepare RDKit molecules for AIMNet2.

    Args:
        rdkit_molecules (list): List of RDKit molecule objects.
        device (str): Torch device.

    Returns:
        dict: AIMNet2 input tensors.
    """
    coord = [mol.GetConformer().GetPositions().tolist() for mol in rdkit_molecules]
    max_n_atoms = max(len(c) for c in coord)

    coordinates = torch.zeros((len(rdkit_molecules), max_n_atoms, 3), device=device)
    atoms = torch.full((len(rdkit_molecules), max_n_atoms), 0, device=device, dtype=torch.long)
    charges = torch.tensor([Chem.GetFormalCharge(mol) for mol in rdkit_molecules], device=device,
                           dtype=torch.long)

    for idx, mol in enumerate(rdkit_molecules):
        n_atoms = len(coord[idx])
        coordinates[idx, :n_atoms] = torch.tensor(coord[idx], device=device)
        atoms[idx, :n_atoms] = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()],
                                            device=device)

    return {"coord": coordinates, "numbers": atoms, "charge": charges}


def prepare_for_aimnet_chunked(rdkit_molecules, device="cpu", chunked=True, min_chunk_size=1000):
    """
    Prepare RDKit molecules for AIMNet2.

    If chunked=True, molecules are grouped into batches with the same number of atoms,
    and the original ordering indices are returned for each batch.
    
    Small chunks (below min_chunk_size) are merged with the closest chunk in terms of molecule size.

    Args:
        rdkit_molecules (list): List of RDKit molecules.
        device (str): Device for tensors ('cpu' or 'cuda').
        chunked (bool): Whether to batch molecules of the same size together.
        min_chunk_size (int): Minimum number of molecules per batch. Small chunks are merged.

    Returns:
        tuple:
            - list of dicts: AIMNet2 input tensors for each batch.
            - list of lists: Indices of molecules in the original list corresponding to each batch.
    """

    if not chunked:
        return [prepare_for_aimnet(rdkit_molecules, device)], [list(range(len(rdkit_molecules)))]

    # Get the number of atoms in each molecule
    idx2num = [(idx, mol.GetNumAtoms()) for idx, mol in enumerate(rdkit_molecules)]
    
    # Group molecules by size
    size_to_molecules = {}
    for idx, num_atoms in idx2num:
        size_to_molecules.setdefault(num_atoms, []).append(idx)

    # Sort chunk sizes by molecule size
    sorted_sizes = sorted(size_to_molecules.keys())

    # Merge small chunks
    merged_chunks = []
    current_chunk = []

    for size in sorted_sizes:
        current_chunk.extend(size_to_molecules[size])
        if len(current_chunk) >= min_chunk_size:
            merged_chunks.append(current_chunk)
            current_chunk = []

    # Add any remaining molecules to the last chunk
    if current_chunk:
        if merged_chunks:
            merged_chunks[-1].extend(current_chunk)
        else:
            merged_chunks.append(current_chunk)

    # Prepare AIMNet input tensors for each batch
    aimnet_batches = []
    for chunk in merged_chunks:
        molecules = [rdkit_molecules[idx] for idx in chunk]
        aimnet_batches.append(prepare_for_aimnet(molecules, device))

    return merged_chunks, aimnet_batches


def check_topology_wrapper(rdkit_mol):
    """
    Wrapper for the check_topology function to validate molecular topology.

    Args:
        rdkit_mol (Chem.Mol): RDKit molecule object.

    Returns:
        bool: True if the topology is valid, False otherwise.
    """
    adjacency_matrix = Chem.GetAdjacencyMatrix(rdkit_mol)
    coordinates = np.array(rdkit_mol.GetConformer().GetPositions().tolist()).reshape(1, -1, 3)
    numbers = np.array([atom.GetAtomicNum() for atom in rdkit_mol.GetAtoms()])

    res = check_topology(adjacency_matrix, numbers, coordinates)

    return bool(res[0])


class Forces(nn.Module):
    def __init__(self, module: nn.Module, x: str = 'coord', y: str = 'energy',
            key_out: str = 'forces'):
        super().__init__()
        self.module = module
        self.x = x
        self.y = y
        self.key_out = key_out

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        data[self.x].requires_grad_(True)
        data = self.module(data)
        y = data[self.y]
        g = torch.autograd.grad([y.sum()], [data[self.x]], create_graph=self.training)[0]
        assert g is not None
        data[self.key_out] = -g
        torch.set_grad_enabled(prev)
        return data


class MoleculeAIMNet2Metrics:
    """
    Compute 3D metrics for molecules, including bond lengths, angles, and torsions.
    """

    def __init__(self, model_path, batchsize, opt_metrics=False, device="cpu", opt_params=None,
                 chunked=False):
        self.model = Forces(torch.jit.load(model_path)).to(device).eval()
        self.opt_metrics = opt_metrics
        self.opt_params = opt_params or {}
        self.device = device
        self.batchsize = batchsize
        self.chunked = chunked

    @torch.no_grad()
    def __call__(self, molecules, reference_molecules=None, return_molecules=False):
        """
        Compute molecular metrics.

        Args:
            molecules (list): List of RDKit molecule objects.

        Returns:
            dict: Computed metrics.
        """
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        valid = [is_valid(mol) for mol in molecules]
        valid_molecules = [mol for mol, val in zip(molecules, valid) if val]

        if reference_molecules is not None:
            assert len(molecules) == len(reference_molecules)
            reference_molecules = [mol for mol, val in zip(reference_molecules, valid) if val]

        aimnet2_idxs, aimnet2_chunks = prepare_for_aimnet_chunked(valid_molecules, device=self.device)


        energy, max_forces = torch.zeros(len(valid_molecules), device=self.device, dtype=torch.double), torch.zeros(len(valid_molecules), device=self.device, dtype=torch.float)

        for chunk, chunk_idxs in zip(aimnet2_chunks, aimnet2_idxs):
            idxs = torch.tensor(chunk_idxs, dtype=torch.long, device=self.device)
            e, f = self.calculate_energy_forces_batched(chunk)
            max_f_norm = (f * (chunk["numbers"] != 0).unsqueeze(-1)).norm(dim=2).max(dim=1).values
            energy[idxs] = e
            max_forces[idxs] = max_f_norm

        metrics = {
            "avg_max_forces": max_forces.mean().item(),
            "median_max_forces": torch.median(max_forces).item(),
        }

        ref_energy = None
        ref_forces = None

        if reference_molecules is not None:
            aimnet2_idxs_ref, aimnet2_chunks_ref = prepare_for_aimnet_chunked(reference_molecules, device=self.device)
            ref_energy = torch.zeros(len(valid_molecules), device=self.device, dtype=torch.double)

            for chunk, chunk_idxs in zip(aimnet2_chunks_ref, aimnet2_idxs_ref):
                idxs = torch.tensor(chunk_idxs, dtype=torch.long, device=self.device)
                e, f = self.calculate_energy_forces_batched(chunk)
                ref_energy[idxs] = e

            ev2kcalpermol = 23.060547830619026
            metrics = {
                "mean_relative_energy": ((energy - ref_energy) * ev2kcalpermol).mean().item(),
                "median_relative_energy": torch.median((energy - ref_energy) * ev2kcalpermol).item(),
            }
            

        if self.opt_metrics:
            start_time = time.time()
            valid_molecules, opt_molecules, res_energy = self.compute_optimized_metrics(aimnet2_chunks, aimnet2_idxs, valid_molecules, energy, metrics,
                                           ref_energy=ref_energy)
            
            end_time = time.time()  # End timing
            metrics["opt_total_time"] = end_time - start_time
            
        if return_molecules:
            if self.opt_metrics:
                return metrics, valid_molecules, opt_molecules, res_energy
            else:
                return metrics, valid_molecules
        else: 
            return metrics

    def calculate_energy_forces_batched(self, aimnet2_batch):
        """
        Calculate forces in smaller batches to optimize memory usage.

        Args:
            aimnet2_batch (dict): Input batch for AIMNet2.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Energy and forces tensors.
        """
        coords = aimnet2_batch["coord"]
        total_energy, total_forces = [], []

        for start_idx in range(0, len(coords), self.batchsize):
            sub_batch = {k: v[start_idx: start_idx + self.batchsize] for k, v in
                         aimnet2_batch.items()}
            out = self.model(sub_batch)
            total_energy.append(out["energy"])
            total_forces.append(out["forces"])

        return torch.cat(total_energy), torch.cat(total_forces)

    def compute_optimized_metrics(self, aimnet2_chunk, aimnet2_idxs, valid_molecules, energy, metrics,
                                  ref_energy=None):
        """
        Compute metrics for optimized geometries.

        Args:
            aimnet2_batch (dict): Input batch.
            valid_molecules (list): List of valid RDKit molecules.
            energy (torch.Tensor): Initial energy values.
            metrics (dict): Dictionary to store metrics.
        """
        opt_molecules = deepcopy(valid_molecules)
        opt_energy = torch.zeros_like(energy)
        opt_converged = torch.zeros_like(energy, dtype=torch.long)
        opt_n_steps = torch.zeros_like(energy, dtype=torch.long)

        for chunk, chunk_idxs in zip(aimnet2_chunk, aimnet2_idxs):
            converged, res_coord, res_energy, _, n_steps = group_opt(
                self.model,
                chunk["coord"],
                chunk["numbers"],
                chunk["charge"],
                device=self.device,
                batchsize=self.batchsize,
                **self.opt_params,
            )

            res_coord = res_coord.cpu().detach().numpy()


            for idx_conf, idx_mol in enumerate(chunk_idxs):
                mol = opt_molecules[idx_mol]
                conformer = mol.GetConformer()
                for i in range(mol.GetNumAtoms()):
                    conformer.SetAtomPosition(i, res_coord[idx_conf, i].tolist())
            idxs = torch.tensor(chunk_idxs, dtype=torch.long)
            opt_converged[idxs] = converged
            opt_energy[idxs] = res_energy
            opt_n_steps[idxs] = n_steps

        pairs = list(zip(valid_molecules, opt_molecules))
        ev2kcalpermol = 23.060547830619026
        bond_diff = compute_distance(pairs, 1, compute_bond_lengths_diff)
        angle_diff = compute_distance(pairs, 2, compute_bond_angles_diff)
        torsion_diff = compute_distance(pairs, 3, compute_torsion_angles_diff)

        topology_mask = torch.tensor([check_topology_wrapper(mol) for mol in opt_molecules],
                                     dtype=torch.bool)
        energy_drop = (energy - opt_energy)[topology_mask] * ev2kcalpermol
        metrics.update({
            "opt_converged": opt_converged.float().mean().item(),
            "opt_steps": opt_n_steps[converged].float().mean().item(),
            "preserved_topology": topology_mask.float().mean().item(),
            "opt_avg_energy_drop": energy_drop.mean().item(),
            "opt_median_energy_drop": torch.median(energy_drop).item(),
            "opt_bond_lengths_diff": bond_diff,
            "opt_bond_angles_diff": angle_diff,
            "opt_dihedrals_diff": torsion_diff,
        })

        if ref_energy is not None:
            metrics["opt_median_relative_energy"] = torch.median(
                (opt_energy - ref_energy)[topology_mask]*ev2kcalpermol).item()
            
            metrics["opt_min_conformers"] = ((opt_energy - ref_energy)[topology_mask]*ev2kcalpermol < 0.1).sum().item() / len(valid_molecules)
            metrics["opt_better_min_conformers"] =  ((opt_energy - ref_energy)[topology_mask]*ev2kcalpermol < -0.1).sum().item() / len(valid_molecules)
        return valid_molecules, opt_molecules, opt_energy

    @staticmethod
    def default_values():
        """Return default metric values."""
        return {
            "forces": 0.0,
            "bond_lengths_diff": 10.0,
            "bond_angles_diff": 10.0,
            "dihedrals_diff": 30.0,
        }
