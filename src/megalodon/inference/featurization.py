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

from typing import Dict, List, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

# Atom type encoder: maps element symbol to integer index.
# Must match full_atom_encoder in src/megalodon/data/data_utils.py exactly.
_ATOM_ENCODER = {
    "H": 0, "B": 1, "C": 2, "N": 3, "O": 4, "F": 5,
    "Al": 6, "Si": 7, "P": 8, "S": 9, "Cl": 10, "As": 11,
    "Br": 12, "I": 13, "Hg": 14, "Bi": 15, "Se": 16,
}


def _add_stereo_bonds(
    mol: Chem.Mol,
    chi_bonds: List[int],
    ez_bonds: Dict,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    from_3D: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add stereochemistry virtual edges to the molecular graph.

    Copied from scripts/sample_conformers.py to avoid circular imports.
    from_3D=False reads stereo from SMILES (no 3D conformer needed).
    """
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
            idx_5 = [n.GetIdx() for n in atom_1.GetNeighbors() if n.GetIdx() not in {idx_2, idx_3}]
            idx_6 = [n.GetIdx() for n in atom_2.GetNeighbors() if n.GetIdx() not in {idx_1, idx_4}]
            inv_stereo = Chem.BondStereo.STEREOE if stereo == Chem.BondStereo.STEREOZ else Chem.BondStereo.STEREOZ
            result.extend([(idx_3, idx_4, ez_bonds[stereo]), (idx_4, idx_3, ez_bonds[stereo])])
            if idx_5:
                result.extend([(idx_5[0], idx_4, ez_bonds[inv_stereo]), (idx_4, idx_5[0], ez_bonds[inv_stereo])])
            if idx_6:
                result.extend([(idx_3, idx_6[0], ez_bonds[inv_stereo]), (idx_6[0], idx_3, ez_bonds[inv_stereo])])
            if idx_5 and idx_6:
                result.extend([(idx_5[0], idx_6[0], ez_bonds[stereo]), (idx_6[0], idx_5[0], ez_bonds[stereo])])

        if bond.GetBeginAtom().HasProp("_CIPCode"):
            chirality = bond.GetBeginAtom().GetProp("_CIPCode")
            neighbors = bond.GetBeginAtom().GetNeighbors()
            if all(n.HasProp("_CIPRank") for n in neighbors):
                sorted_neighbors = sorted(neighbors, key=lambda x: int(x.GetProp("_CIPRank")), reverse=True)
                sorted_neighbors = [a.GetIdx() for a in sorted_neighbors]
                a, b, c = sorted_neighbors[:3] if chirality == "R" else sorted_neighbors[:3][::-1]
                d = sorted_neighbors[-1]
                result.extend([
                    (a, d, chi_bonds[0]), (b, d, chi_bonds[0]), (c, d, chi_bonds[0]),
                    (d, a, chi_bonds[0]), (d, b, chi_bonds[0]), (d, c, chi_bonds[0]),
                    (b, a, chi_bonds[1]), (c, b, chi_bonds[1]), (a, c, chi_bonds[1]),
                ])

    if not result:
        return edge_index, edge_attr
    new_edge_index = torch.tensor([[i, j] for i, j, _ in result], dtype=torch.long).T
    new_edge_attr = torch.tensor([b for _, _, b in result], dtype=torch.uint8)
    edge_index = torch.cat([edge_index, new_edge_index], dim=1)
    edge_attr = torch.cat([edge_attr, new_edge_attr])
    return edge_index, edge_attr


def _mol_to_pyg(mol: Chem.Mol, smiles: str) -> Data:
    """Convert RDKit molecule (with Hs, no 3D conformer) to PyG Data.

    Zero-filled coordinates — the diffusion model replaces them during sampling.
    Stereochemistry is read from SMILES (from_3D=False).
    """
    Chem.SanitizeMol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)

    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.to(torch.uint8)

    pos = torch.zeros((mol.GetNumAtoms(), 3), dtype=torch.float32)

    atom_types = torch.tensor(
        [_ATOM_ENCODER[atom.GetSymbol()] for atom in mol.GetAtoms()],
        dtype=torch.uint8,
    )
    charges = torch.tensor(
        [atom.GetFormalCharge() for atom in mol.GetAtoms()],
        dtype=torch.int8,
    )

    chi_bonds = [7, 8]
    ez_bonds = {Chem.BondStereo.STEREOE: 5, Chem.BondStereo.STEREOZ: 6}
    edge_index, edge_attr = _add_stereo_bonds(
        mol, chi_bonds, ez_bonds, edge_index, edge_attr, from_3D=False
    )

    return Data(
        x=atom_types,
        edge_index=edge_index,
        edge_attr=edge_attr.to(torch.uint8),
        pos=pos,
        charges=charges,
        smiles=smiles,
        mol=mol,
        chemblid=mol.GetProp("_Name") if mol.HasProp("_Name") else "",
    )


def build_data_list(
    valid_entries: List[Tuple[int, str, Chem.Mol]],
    n_confs_per_mol: List[int],
) -> Tuple[List[Data], List[int], List[int]]:
    """Build a flat list of PyG Data objects with parallel identity tracking lists.

    Args:
        valid_entries: list of (smiles_idx, smiles, rdkit_mol_with_hs)
        n_confs_per_mol: list of int, one per entry in valid_entries

    Returns:
        (data_list, source_smiles_indices, conformer_indices)
        All three lists are parallel — index i refers to the same item across them.
    """
    data_list: List[Data] = []
    source_smiles_indices: List[int] = []
    conformer_indices: List[int] = []

    for (smiles_idx, smiles, mol), n_confs in zip(valid_entries, n_confs_per_mol):
        base_data = _mol_to_pyg(mol, smiles)
        for conf_idx in range(n_confs):
            data_list.append(base_data.clone())
            source_smiles_indices.append(smiles_idx)
            conformer_indices.append(conf_idx)

    return data_list, source_smiles_indices, conformer_indices


def debatch_conformers(
    generated_mols: List,
    source_smiles_indices: List[int],
    smiles_list: List[str],
) -> Dict[str, List]:
    """Reconstruct {smiles: [mol, ...]} from flat generated list + parallel tracking.

    Args:
        generated_mols: flat list, parallel to source_smiles_indices
        source_smiles_indices: source_smiles_indices[i] = index into smiles_list
        smiles_list: original list of SMILES (all, including failed ones)

    Returns:
        {smiles_string: [mol, ...]} — keys only for successfully processed SMILES
    """
    result: Dict[str, List] = {}
    for mol, src_idx in zip(generated_mols, source_smiles_indices):
        smiles = smiles_list[src_idx]
        if smiles not in result:
            result[smiles] = []
        result[smiles].append(mol)
    return result


def _convert_coords_to_np(out):
    """
    Converts the output dictionary containing 'x' (coordinates) and 'batch' (molecule indices)
    into a list of NumPy arrays, where each entry represents coordinates for one molecule.

    Parameters:
        out (dict): Dictionary containing:
            - 'x' (torch.Tensor): Tensor of atomic coordinates (N, 3)
            - 'batch' (torch.Tensor): Tensor indicating molecule index for each atom

    Returns:
        List[np.ndarray]: List where each element is a NumPy array (M, 3) for a molecule.
    """
    coords_list = []

    x = out["x"].cpu().numpy()  # Convert tensor to NumPy (N, 3)
    batch = out["batch"].cpu().numpy()  # Convert tensor to NumPy (N,)

    unique_mols = np.unique(batch)  # Get unique molecule indices

    for mol_id in unique_mols:
        coords_list.append(x[batch == mol_id])  # Select coordinates for each molecule

    return coords_list


def _write_coords_to_mol(mol, coord):
    """
    Embeds 3D coordinates into an RDKit molecule and assigns stereochemistry.
    """

    # Deserialize RDKit molecule
    rdkit_mol = Chem.Mol(mol)

    rdkit_mol.RemoveAllConformers()
    conf = Chem.Conformer(rdkit_mol.GetNumAtoms())

    coords = np.asarray(coord)

    for i in range(rdkit_mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))

    rdkit_mol.AddConformer(conf)

    return rdkit_mol
