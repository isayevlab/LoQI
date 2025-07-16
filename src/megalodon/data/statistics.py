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

import numpy as np
import pickle
import torch


def save_pickle(array, path):
    with open(path, "wb") as f:
        pickle.dump(array, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


class Statistics:
    def __init__(
            self,
            num_nodes,
            atom_types,
            bond_types,
            charge_types,
            valencies,
            bond_lengths=None,
            bond_angles=None,
            dihedrals=None,
            is_in_ring=None,
            is_aromatic=None,
            hybridization=None,
            force_norms=None,
            smiles=None
    ):
        self.num_nodes = num_nodes
        self.atom_types = atom_types
        self.bond_types = bond_types
        self.charge_types = charge_types
        self.valencies = valencies
        self.bond_lengths = bond_lengths
        self.bond_angles = bond_angles
        self.dihedrals = dihedrals
        self.is_in_ring = is_in_ring
        self.is_aromatic = is_aromatic
        self.hybridization = hybridization
        self.force_norms = force_norms
        self.smiles = smiles

    @classmethod
    def load_statistics(cls, statistics_dir, split_name):
        h = "h"
        return cls(
            num_nodes=load_pickle(f"{statistics_dir}/{split_name}_n_{h}.pickle"),
            atom_types=torch.from_numpy(np.load(f"{statistics_dir}/{split_name}_atom_types_{h}.npy")),
            bond_types=torch.from_numpy(np.load(f"{statistics_dir}/{split_name}_bond_types_{h}.npy")),
            charge_types=torch.from_numpy(np.load(f"{statistics_dir}/{split_name}_charges_{h}.npy")),
            valencies=load_pickle(f"{statistics_dir}/{split_name}_valency_{h}.pickle"),
            is_aromatic=torch.from_numpy(np.load(f"{statistics_dir}/{split_name}_is_aromatic_{h}.npy")).float(),
            is_in_ring=torch.from_numpy(np.load(f"{statistics_dir}/{split_name}_is_in_ring_{h}.npy")).float(),
            hybridization=torch.from_numpy(np.load(f"{statistics_dir}/{split_name}_hybridization_{h}.npy")).float(),
            bond_lengths=load_pickle(f"{statistics_dir}/{split_name}_bond_lengths_{h}.pickle"),
            bond_angles=load_pickle(f"{statistics_dir}/{split_name}_angles_{h}.pickle"),
            dihedrals=load_pickle(f"{statistics_dir}/{split_name}_dihedrals_{h}.pickle"),
            smiles=load_pickle(f"{statistics_dir}/{split_name}_smiles.pickle")
        )
