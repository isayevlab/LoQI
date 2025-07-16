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

from rdkit import Chem
from rdkit.Chem import rdMolTransforms


def generate_canonical_key(*components):
    """
    Generate a canonical key for any molecular component (atoms, bonds).
    This works for angles, bond lengths, and torsions.
    """
    key1 = tuple(components)
    key2 = tuple(reversed(components))
    return min(key1, key2)


def compute_bond_angles(rdkit_mol):
    bond_angles = {}
    conf = rdkit_mol.GetConformer()

    for atom in rdkit_mol.GetAtoms():
        neighbors = atom.GetNeighbors()
        if len(neighbors) < 2:
            continue

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                idx1, idx2, idx3 = neighbors[i].GetIdx(), atom.GetIdx(), neighbors[j].GetIdx()
                atom1_type, atom2_type, atom3_type = rdkit_mol.GetAtomWithIdx(
                    idx1).GetAtomicNum(), rdkit_mol.GetAtomWithIdx(
                    idx2).GetAtomicNum(), rdkit_mol.GetAtomWithIdx(idx3).GetAtomicNum()
                bond_type_1 = int(rdkit_mol.GetBondBetweenAtoms(idx1, idx2).GetBondType())
                bond_type_2 = int(rdkit_mol.GetBondBetweenAtoms(idx2, idx3).GetBondType())

                angle_init = rdMolTransforms.GetAngleDeg(conf, idx1, idx2, idx3)

                key = generate_canonical_key(atom1_type, bond_type_1, atom2_type, bond_type_2,
                                             atom3_type)
                if key not in bond_angles:
                    bond_angles[key] = [[], 0]
                bond_angles[key][0].append(angle_init)
                bond_angles[key][1] += 1

    return bond_angles


def compute_bond_lengths(rdkit_mol):
    bond_lengths = {}

    conf = rdkit_mol.GetConformer()

    for bond in rdkit_mol.GetBonds():
        idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        atom1_type, atom2_type = rdkit_mol.GetAtomWithIdx(
            idx1).GetAtomicNum(), rdkit_mol.GetAtomWithIdx(idx2).GetAtomicNum()
        bond_type_numeric = int(bond.GetBondType())
        length = rdMolTransforms.GetBondLength(conf, idx1, idx2)
        key = generate_canonical_key(atom1_type, bond_type_numeric, atom2_type)
        if key not in bond_lengths:
            bond_lengths[key] = [[], 0]
        bond_lengths[key][0].append(length)
        bond_lengths[key][1] += 1
    return bond_lengths


def compute_torsion_angles(rdkit_mol):
    torsionSmarts = "[!$(*#*)&!D1]~[!$(*#*)&!D1]"
    torsion_query = Chem.MolFromSmarts(torsionSmarts)

    torsion_angles = {}

    init_conf = rdkit_mol.GetConformer()

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

                atom1_type, atom2_type, atom3_type, atom4_type = rdkit_mol.GetAtomWithIdx(
                    idx1).GetAtomicNum(), rdkit_mol.GetAtomWithIdx(
                    idx2).GetAtomicNum(), rdkit_mol.GetAtomWithIdx(
                    idx3).GetAtomicNum(), rdkit_mol.GetAtomWithIdx(idx4).GetAtomicNum()
                bond_type_1 = int(b1.GetBondType())
                bond_type_2 = int(bond.GetBondType())
                bond_type_3 = int(b2.GetBondType())

                angle = rdMolTransforms.GetDihedralDeg(init_conf, idx1, idx2, idx3, idx4)
                key = generate_canonical_key(atom1_type, bond_type_1, atom2_type, bond_type_2,
                                             atom3_type, bond_type_3, atom4_type)

                if key not in torsion_angles:
                    torsion_angles[key] = [[], 0]
                torsion_angles[key][0].append(angle)
                torsion_angles[key][1] += 1

    return torsion_angles
