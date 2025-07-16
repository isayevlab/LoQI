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
from rdkit import Chem
from scipy.stats import wasserstein_distance
from torchmetrics import MeanMetric

from megalodon.metrics.geometry import (
    compute_bond_lengths,
    compute_bond_angles,
    compute_torsion_angles,
)


def is_valid(mol, verbose=False):
    """
    Check if a molecule is valid based on:
    1. It should consist of only one fragment.
    2. It should pass sanitization without errors.

    Args:
        mol: RDKit molecule object.
        verbose (bool): Print error messages if validation fails.

    Returns:
        bool: True if the molecule is valid, False otherwise.
    """
    mol = Chem.Mol(mol)
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


def collect_geometry(mols, compute_function, preserve_aromatic=True):
    """
    Compute geometry metrics for a set of molecules using a specified function.

    Args:
        mols (list): List of RDKit molecule objects.
        compute_function (callable): Function to compute geometry metrics.
        preserve_aromatic (bool): Whether to preserve aromatic bond information.
                                 Should be True to match GEOM processing pipeline.

    Returns:
        dict: Aggregated geometry metrics.
    """
    diff_sums = {}
    results = []

    for mol in mols:
        if is_valid(mol):
            mol = Chem.Mol(mol)
            Chem.SanitizeMol(mol)
            if not preserve_aromatic:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            result = compute_function(mol)
            results.append(result)

    for result in results:
        for key, (diff_list, count) in result.items():
            if key not in diff_sums:
                diff_sums[key] = [[], 0]
            diff_sums[key][0].extend(diff_list)
            diff_sums[key][1] += count

    return diff_sums


def aggregate_dict(dct, agg_idx):
    """
    Aggregate a dictionary of geometry metrics by a specific key index.

    Args:
        dct (dict): Dictionary containing geometry metrics.
        agg_idx (int): Index to aggregate by.

    Returns:
        dict: Aggregated dictionary.
    """
    res = {}
    for k, v in dct.items():
        if k[agg_idx] not in res:
            res[k[agg_idx]] = [[], 0]
        res[k[agg_idx]][0].extend(v[0])
        res[k[agg_idx]][1] += v[1]
    return res


def wasserstein(data, result, support="joint", bins=1000):
    """
    Compute the Wasserstein distance between two distributions.

    Args:
        data (list): Reference data distribution.
        result (list): Resulting data distribution.
        support (str or tuple): Range to consider ("joint", "data", or custom range).
        bins (int): Number of bins for histogram.

    Returns:
        float: Wasserstein distance.
    """
    if support == "joint":
        x_min, x_max = min(min(data), min(result)), max(max(data), max(result))
    elif support == "data":
        x_min, x_max = min(data), max(data)
    else:
        x_min, x_max = support

    y_data, x_data = np.histogram(data, bins=bins, range=(x_min, x_max), density=True)
    y_res, x_res = np.histogram(result, bins=bins, range=(x_min, x_max), density=True)
    y_res[np.isnan(y_res)] = 0.

    return wasserstein_distance(
        (x_data[:-1] + x_data[1:]) / 2, (x_res[:-1] + x_res[1:]) / 2, y_data, y_res
    )


def compute_distance(rdkit_molecules, dataset_values, agg_idx, compute_function, support="joint",
        bins=1000, preserve_aromatic=True):
    """
    Compute the weighted Wasserstein distance between molecule geometries.

    Args:
        rdkit_molecules (list): List of RDKit molecule objects.
        dataset_values (dict): Reference geometry statistics.
        agg_idx (int): Aggregation key index.
        compute_function (callable): Geometry computation function.
        support (str or tuple): Range of support for Wasserstein distance.
        bins (int): Number of bins for histogram.
        preserve_aromatic (bool): Whether to preserve aromatic bond information.

    Returns:
        float: Weighted Wasserstein distance.
    """
    result_dict = collect_geometry(rdkit_molecules, compute_function, preserve_aromatic=preserve_aromatic)
    agg_dataset = aggregate_dict(dataset_values, agg_idx)
    agg_res = aggregate_dict(result_dict, agg_idx)

    tot_n = sum(v[1] for v in agg_dataset.values())
    weights = {k: v[1] / tot_n for k, v in agg_dataset.items()}

    res_distance = 0

    for k in agg_dataset:
        if k in agg_res and agg_res[k][1] > 1:
            res_distance += wasserstein(agg_dataset[k][0], agg_res[k][0], support=support,
                                        bins=bins) * weights[k]
    return res_distance


class Molecule3DMetrics:
    """
    Class to compute 3D metrics for molecules, including bond lengths, angles, and optionally dihedrals.
    """

    def __init__(self, dataset_info, device="cpu", test=False, preserve_aromatic=False):
        self.bond_lengths_w1 = MeanMetric().to(device)
        self.angles_w1 = MeanMetric().to(device)    
        self.dihedrals_w1 = MeanMetric().to(device)
        self.statistics = dataset_info["statistics"]
        self.test = test
        self.preserve_aromatic = preserve_aromatic


    def reset(self):
        """Reset all metrics."""
        self.bond_lengths_w1.reset()
        self.angles_w1.reset()
        if self.dihedrals_w1:
            self.dihedrals_w1.reset()

    def __call__(self, molecules):
        """
        Compute metrics for a batch of molecules.

        Args:
            molecules (list): List of molecule objects.

        Returns:
            dict: Computed metrics.
        """
        stats = self.statistics
        if stats is None:
            return {}
        bond_lengths = compute_distance(molecules, stats.bond_lengths, 1, compute_bond_lengths,
                                        support="joint", preserve_aromatic=self.preserve_aromatic)
        bond_angles = compute_distance(molecules, stats.bond_angles, 2, compute_bond_angles,
                                       support=(0, 360), bins=360, preserve_aromatic=self.preserve_aromatic)
        torsions = compute_distance(molecules, stats.dihedrals, 3, compute_torsion_angles,
                                    support=(0, 360), bins=360, preserve_aromatic=self.preserve_aromatic)

        self.bond_lengths_w1(bond_lengths)
        self.angles_w1(bond_angles)
        if self.dihedrals_w1:
            self.dihedrals_w1(torsions)

        metrics = {
            "bond_lengths": self.bond_lengths_w1.compute().item(),
            "bond_angles": self.angles_w1.compute().item(),
            "dihedrals": self.dihedrals_w1.compute().item()
        }

        return metrics

    @staticmethod
    def default_values():
        """Return default metric values."""
        return {"bond_lengths": 10.0, "bond_angles": 10.0, "dihedrals": 10.0}
