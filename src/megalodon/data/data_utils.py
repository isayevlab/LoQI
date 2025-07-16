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


import os
import pickle
from glob import glob

import rdkit

# import rmsd
import torch
from openbabel import pybel
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.utils import sort_edge_index, subgraph


RDLogger.DisableLog("rdApp.*")

x_map = {
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
    "hybridization": [
        rdkit.Chem.rdchem.HybridizationType.UNSPECIFIED,
        rdkit.Chem.rdchem.HybridizationType.S,
        rdkit.Chem.rdchem.HybridizationType.SP,
        rdkit.Chem.rdchem.HybridizationType.SP2,
        rdkit.Chem.rdchem.HybridizationType.SP3,
        rdkit.Chem.rdchem.HybridizationType.SP2D,
        rdkit.Chem.rdchem.HybridizationType.SP3D,
        rdkit.Chem.rdchem.HybridizationType.SP3D2,
        rdkit.Chem.rdchem.HybridizationType.OTHER,
    ],
}

full_atom_encoder = {
    "H": 0,
    "B": 1,
    "C": 2,
    "N": 3,
    "O": 4,
    "F": 5,
    "Al": 6,
    "Si": 7,
    "P": 8,
    "S": 9,
    "Cl": 10,
    "As": 11,
    "Br": 12,
    "I": 13,
    "Hg": 14,
    "Bi": 15,
    "Se": 16
}


def mol_to_torch_geometric(
    mol,
    atom_encoder,
    smiles,
    remove_hydrogens: bool = False,
    cog_proj: bool = True,
    **kwargs,
):
    if remove_hydrogens:
        # mol = Chem.RemoveAllHs(mol)
        mol = Chem.RemoveHs(mol)  # only remove (explicit) hydrogens attached to molecular graph
        Chem.Kekulize(mol, clearAromaticFlags=True)

    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    if remove_hydrogens:
        assert max(bond_types) != 4
    edge_attr = bond_types.long()

    pos = torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    if cog_proj:
        pos = pos - torch.mean(pos, dim=0, keepdim=True)
    atom_types = []
    all_charges = []
    is_aromatic = []
    is_in_ring = []
    sp_hybridization = []

    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
        all_charges.append(atom.GetFormalCharge())  # TODO: check if implicit Hs should be kept
        is_aromatic.append(x_map["is_aromatic"].index(atom.GetIsAromatic()))
        is_in_ring.append(x_map["is_in_ring"].index(atom.IsInRing()))
        sp_hybridization.append(x_map["hybridization"].index(atom.GetHybridization()))

    atom_types = torch.Tensor(atom_types).long()
    all_charges = torch.Tensor(all_charges).long()

    is_aromatic = torch.Tensor(is_aromatic).long()
    is_in_ring = torch.Tensor(is_in_ring).long()
    hybridization = torch.Tensor(sp_hybridization).long()

    additional = {}
    if "wbo" in kwargs:
        wbo = torch.Tensor(kwargs["wbo"])[edge_index[0], edge_index[1]].float()
        additional["wbo"] = wbo
    if "mulliken" in kwargs:
        mulliken = torch.Tensor(kwargs["mulliken"]).float()
        additional["mulliken"] = mulliken
    if "grad" in kwargs:
        grad = torch.Tensor(kwargs["grad"]).float()
        additional["grad"] = grad

    data = Data(
        x=atom_types,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        charges=all_charges,
        smiles=smiles,
        is_aromatic=is_aromatic,
        is_in_ring=is_in_ring,
        hybridization=hybridization,
        mol=mol,
        **additional,
    )

    return data


# in case the rdkit.molecule has explicit hydrogens, the number of attached hydrogens to heavy atoms are not saved
def remove_hydrogens(data: Data):
    to_keep = data.x > 0
    new_edge_index, new_edge_attr = subgraph(
        to_keep,
        data.edge_index,
        data.edge_attr,
        relabel_nodes=True,
        num_nodes=len(to_keep),
    )
    new_pos = data.pos[to_keep] - torch.mean(data.pos[to_keep], dim=0)

    newdata = Data(
        x=data.x[to_keep] - 1,  # Shift onehot encoding to match atom decoder
        pos=new_pos,
        charges=data.charges[to_keep],
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        mol=data.mol,
    )

    if hasattr(data, "is_aromatic"):
        newdata["is_aromatic"] = data.get("is_aromatic")[to_keep]
    if hasattr(data, "is_in_ring"):
        newdata["is_in_ring"] = data.get("is_in_ring")[to_keep]
    if hasattr(data, "hybridization"):
        newdata["hybridization"] = data.get("hybridization")[to_keep]

    return newdata


def save_pickle(array, path):
    with open(path, "wb") as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def write_xyz_file(coords, atom_types, filename):
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, "w") as f:
        f.write(out)


def write_xyz_file_from_batch(
    pos,
    atoms,
    batch,
    atom_decoder=None,
    pos_pocket=None,
    atoms_pocket=None,
    batch_pocket=None,
    joint_traj=False,
    path="/scratch1/e3moldiffusion/logs/crossdocked",
    i=0,
):
    if not os.path.exists(path):
        os.makedirs(path)

    atomsxmol = batch.bincount()
    num_atoms_prev = 0
    for k, num_atoms in enumerate(atomsxmol):
        save_dir = os.path.join(path, f"batch_{k}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ats = torch.argmax(atoms[num_atoms_prev : num_atoms_prev + num_atoms], dim=1)
        types = [atom_decoder[int(a)] for a in ats]
        positions = pos[num_atoms_prev : num_atoms_prev + num_atoms]
        write_xyz_file(positions, types, os.path.join(save_dir, f"mol_{i}.xyz"))

        num_atoms_prev += num_atoms

    if joint_traj:
        atomsxmol = batch.bincount()
        atomsxmol_pocket = batch_pocket.bincount()
        num_atoms_prev = 0
        num_atoms_prev_pocket = 0
        for k, (num_atoms, num_atoms_pocket) in enumerate(zip(atomsxmol, atomsxmol_pocket)):
            save_dir = os.path.join(path, f"batch_{k}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            ats = torch.argmax(atoms[num_atoms_prev : num_atoms_prev + num_atoms], dim=1)
            ats_pocket = torch.argmax(
                atoms_pocket[num_atoms_prev_pocket : num_atoms_prev_pocket + num_atoms_pocket],
                dim=1,
            )
            types = [atom_decoder[int(a)] for a in ats]
            types_pocket = ["B" for _ in range(len(ats_pocket))]  # [atom_decoder[int(a)] for a in ats_pocket]
            positions = pos[num_atoms_prev : num_atoms_prev + num_atoms]
            positions_pocket = pos_pocket[num_atoms_prev_pocket : num_atoms_prev_pocket + num_atoms_pocket]

            types_joint = types + types_pocket
            positions_joint = torch.cat([positions, positions_pocket], dim=0)

            write_xyz_file(
                positions_joint,
                types_joint,
                os.path.join(save_dir, f"lig_pocket_{i}.xyz"),
            )

            num_atoms_prev += num_atoms
            num_atoms_prev_pocket += num_atoms_pocket


def get_key(fp):
    filename = os.path.splitext(os.path.basename(fp))[0]
    int_part = filename.split("_")[-1]
    return int(int_part)


def write_trajectory_as_xyz(
    molecules,
    path,
    strict=True,
):
    try:
        os.makedirs(path)
    except OSError:
        pass

    for i, mol in enumerate(molecules):
        rdkit_mol = mol.rdkit_mol
        valid = rdkit_mol is not None and mol.compute_validity(rdkit_mol, strict=strict) is not None
        if valid:
            files = sorted(glob(os.path.join(path, f"batch_{i}/mol_*.xyz")), key=get_key)
            traj_path = os.path.join(path, f"trajectory_{i}.xyz")
            for j, file in enumerate(files):
                with open(file, "r") as f:
                    lines = f.readlines()

                with open(traj_path, "a") as file:
                    for line in lines:
                        file.write(line)
                    if j == len(files) - 1:  ####write the last timestep 10x for better visibility
                        for _ in range(10):
                            for line in lines:
                                file.write(line)


def get_rdkit_mol(fname_xyz):
    mol = next(pybel.readfile("xyz", fname_xyz))
    mol = Chem.MolFromPDBBlock(
        molBlock=mol.write(format="pdb"),
        sanitize=False,
        removeHs=False,
        proximityBonding=True,
    )
    # assert len(Chem.GetMolFrags(mol)) == 2
    return mol


# def calc_rmsd(mol1, mol2):
#     U = rmsd.kabsch(mol1, mol2)
#     mol1 = np.dot(mol1, U)
#     return rmsd.rmsd(mol1, mol2)


def create_bond_graph(data, atom_encoder):
    mol = data.mol
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.long()

    torch.tensor(mol.GetConformers()[0].GetPositions()).float()
    # assert calc_rmsd(pos.numpy(), data.pos.numpy()) < 1.0e-3

    atom_types = []
    all_charges = []
    for atom in mol.GetAtoms():
        atom_types.append(atom_encoder[atom.GetSymbol()])
        all_charges.append(atom.GetFormalCharge())
    atom_types = torch.Tensor(atom_types).long()
    assert (atom_types == data.x).all()

    all_charges = torch.Tensor(all_charges).long()
    data.charges = all_charges

    # edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    # E = to_dense_adj(
    #     edge_index=edge_index,
    #     batch=torch.zeros_like(atom_types),
    #     edge_attr=edge_attr,
    #     max_num_nodes=len(atom_types),
    # )
    # diag_mask = ~torch.eye(5, dtype=torch.bool)
    # E = F.one_hot(E, num_classes=5).float() * diag_mask
    data.bond_index = edge_index
    data.bond_attr = edge_attr

    data = fully_connected_edge_idx(data=data, without_self_loop=True)

    return data


def fully_connected_edge_idx(data: Data, without_self_loop: bool = True):
    N = data.pos.size(0)
    row = torch.arange(N, dtype=torch.long)
    col = torch.arange(N, dtype=torch.long)
    row = row.view(-1, 1).repeat(1, N).view(-1)
    col = col.repeat(N)
    fc_edge_index = torch.stack([row, col], dim=0)
    if without_self_loop:
        mask = fc_edge_index[0] != fc_edge_index[1]
        fc_edge_index = fc_edge_index[:, mask]

    fc_edge_index = sort_edge_index(fc_edge_index, sort_by_row=False, num_nodes=N)
    data.fc_edge_index = fc_edge_index

    return data


def atom_type_config(dataset: str = "qm9"):
    if dataset == "qm9":
        mapping = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    elif dataset == "aqm":
        mapping = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4, "P": 5, "S": 6, "Cl": 7}
    elif dataset == "fullerene":
        mapping = {"H": 0, "C": 1, "N": 2, "Cl": 3}
    elif dataset == "drugs":
        mapping = {
            "H": 0,
            "B": 1,
            "C": 2,
            "N": 3,
            "O": 4,
            "F": 5,
            "Al": 6,
            "Si": 7,
            "P": 8,
            "S": 9,
            "Cl": 10,
            "As": 11,
            "Br": 12,
            "I": 13,
            "Hg": 14,
            "Bi": 15,
        }
    else:
        raise ValueError("Dataset not found!")
    return mapping
