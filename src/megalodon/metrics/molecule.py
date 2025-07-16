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


import torch
from rdkit import Chem
from rdkit.Geometry import Point3D


class Molecule:
    def __init__(self, atom_types, bond_types, positions, dataset_info, charges=None, device="cpu", silent=True):
        """
        Initialize the Molecule class.

        Args:
            atom_types: List or tensor of atom types.
            bond_types: List or tensor of bond types.
            positions: List or tensor of atomic positions.
            dataset_info: Dictionary containing dataset information.
            charges: List or tensor of atomic charges (default: None).
            device: Device to use for tensor operations (default: "cpu").
            silent: If True, suppress error messages (default: True).
        """
        self.silent = silent

        if not torch.is_tensor(atom_types):
            atom_types = torch.tensor(atom_types, device=device)

        if not torch.is_tensor(bond_types):
            bond_types = torch.tensor(bond_types, device=device)

        if not torch.is_tensor(positions):
            positions = torch.tensor(positions, device=device)

        if charges is not None and not torch.is_tensor(charges):
            charges = torch.tensor(charges, device=device)
        elif charges is None:
            charges = torch.zeros_like(atom_types)
        self.num_nodes = len(atom_types)
        self.atom_types = atom_types.long()
        self.bond_types = bond_types.long()
        self.positions = positions
        self.charges = charges.long()
        self.dataset_info = dataset_info
        self.raw_rdkit_mol = self.build_rdkit_mol()
        mol = Chem.Mol(self.raw_rdkit_mol)
        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            if not self.silent:
                print(f"Error sanitizing molecule: {e}")
        self.rdkit_mol = mol

    def build_rdkit_mol(self):
        """
        Build an RDKit molecule from the current attributes.

        Returns:
            RDKit molecule object.
        """
        mol = Chem.RWMol()
        atom_decoder = self.dataset_info["atom_decoder"]
        for i, atom_type in enumerate(self.atom_types):
            try:
                atom = Chem.Atom(atom_decoder[atom_type.item()])
                atom.SetFormalCharge(self.charges[i].item())
                mol.AddAtom(atom)
            except Exception as e:
                if not self.silent:
                    print(f"Error adding atom: {e}")

        bond_types = torch.triu(self.bond_types, diagonal=1)
        bond_dict = [
            None,
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
        for i in range(bond_types.size(0)):
            for j in range(i + 1, bond_types.size(1)):
                bond_type = bond_types[i, j].item()
                if bond_type > 0:
                    try:
                        mol.AddBond(i, j, bond_dict[bond_type])
                    except Exception as e:
                        if not self.silent:
                            print(f"Error adding bond: {e}")

        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            try:
                pos = self.positions[i].cpu().numpy()
                conf.SetAtomPosition(i, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
            except Exception as e:
                if not self.silent:
                    print(f"Error setting atom position: {e}")
        mol.AddConformer(conf)
        mol = mol.GetMol()
        return mol


def get_molecules(mol_dict, dataset_info):
    """
    Convert a dictionary of molecular data into a list of Molecule objects.

    Args:
        mol_dict: Dictionary containing molecular data.
        dataset_info: Dictionary containing dataset information.

    Returns:
        List of Molecule objects.
    """
    batch = mol_dict["batch"]
    edge_index = mol_dict["edge_index"]
    edge_attr = mol_dict["edge_attr"]
    h = mol_dict["h"]
    x = mol_dict["x"]
    if "charges" in mol_dict:
        charges = mol_dict["charges"]
    else:
        charges = torch.zeros_like(h)
    molecule_list = []
    for idx in torch.unique(batch):
        idx_mask = batch == idx
        atom_types = h[idx_mask]
        positions = x[idx_mask]
        ch = charges[idx_mask]

        # Create bond matrix
        edge_mask = (edge_index[0] >= idx_mask.nonzero(as_tuple=True)[0].min()) & (
            edge_index[0] <= idx_mask.nonzero(as_tuple=True)[0].max()
        )
        bonds = edge_attr[edge_mask]
        bond_indices = edge_index[:, edge_mask]

        # Adjust bond indices to local molecule
        local_bond_indices = bond_indices - bond_indices[0].min()

        bond_matrix = torch.zeros((len(atom_types), len(atom_types)), dtype=torch.long, device=atom_types.device)
        for src, dst, bond in zip(local_bond_indices[0], local_bond_indices[1], bonds):
            bond_matrix[src, dst] = bond
            bond_matrix[dst, src] = bond

        molecule = Molecule(
            atom_types=atom_types,
            bond_types=bond_matrix,
            positions=positions,
            charges=ch,
            dataset_info=dataset_info,
        )
        molecule_list.append(molecule)

    return molecule_list
