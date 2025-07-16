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


from copy import deepcopy

import torch
from lightning import pytorch as pl

from megalodon.metrics.molecule import get_molecules
from megalodon.metrics.molecule_metrics_2d import Molecule2DMetrics
from megalodon.metrics.molecule_metrics_3d import Molecule3DMetrics
from megalodon.metrics.molecule_novelty_similarity import MoleculeTrainDataMetrics
from megalodon.metrics.molecule_stability_2d import Molecule2DStability


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

full_atom_decoder = dict(map(reversed, full_atom_encoder.items()))


class MoleculeEvaluationCallback(pl.Callback):
    """
    Callback for evaluating generated molecules on various metrics.

    Args:
        n_graphs (int): Number of molecules to generate for evaluation.
        batch_size (int): Batch size for molecule generation.
        timesteps (Optional[int]): Number of timesteps for sampling.
        train_smiles (Optional[List[str]]): Training dataset SMILES strings for calculating similarity and novelty.
        statistics (Optional[Dict]): Precomputed dataset statistics.
        compute_2D_metrics (bool): Whether to compute 2D metrics.
        compute_3D_metrics (bool): Whether to compute 3D metrics.
        compute_dihedrals (bool): Whether to compute dihedral angles in 3D metrics.
        compute_train_data_metrics (bool): Whether to compute train data metrics (similarity and novelty).
        preserve_aromatic (bool): Whether to preserve aromatic bond information in 3D metrics.
                                 Should be True for GEOM datasets, False for QM9.
    """

    def __init__(
            self,
            n_graphs=500,
            batch_size=100,
            timesteps=None,
            train_smiles=None,
            statistics=None,
            compute_2D_metrics=True,
            compute_3D_metrics=True,
            compute_train_data_metrics=True,
            compute_energy_metrics=True,
            energy_metrics_args=None,
            scale_coords=None,
            preserve_aromatic=True,  # Default to True for GEOM compatibility
    ):
        super().__init__()
        self.n_graphs = n_graphs
        self.batch_size = batch_size
        self.full_atom_decoder = full_atom_decoder
        self.timesteps = timesteps
        if statistics is not None:
            train_smiles = statistics.smiles
        self.train_smiles = train_smiles
        self.dataset_info = {
            "atom_decoder": full_atom_decoder,
            "atom_encoder": full_atom_encoder,
            "statistics": statistics,
        }
        self.compute_2D_metrics = compute_2D_metrics
        self.compute_3D_metrics = compute_3D_metrics
        self.compute_train_data_metrics = compute_train_data_metrics
        self.compute_energy_metrics = compute_energy_metrics
        self.energy_metrics_args = energy_metrics_args
        self.scale_coords = 1.0
        if scale_coords is not None:
            self.scale_coords = scale_coords
        self.preserve_aromatic = preserve_aromatic

    def gather_default_values(self):
        """
        Gather default values for metrics when evaluation fails.

        Returns:
            Dict: Default values for each metric.
        """
        defaults = {}
        defaults.update(Molecule2DStability.default_values())
        if self.compute_2D_metrics:
            defaults.update(Molecule2DMetrics.default_values())
        if self.compute_3D_metrics:
            defaults.update(Molecule3DMetrics.default_values())
        if self.compute_train_data_metrics:
            defaults.update(MoleculeTrainDataMetrics.default_values())
        return defaults

    def evaluate_molecules(self, pl_module, trainer=None, return_molecules=False):
        """
        Evaluate generated molecules on specified metrics.

        Args:
            trainer: PyTorch Lightning trainer instance.
            pl_module: PyTorch Lightning module instance.

        Returns:
            Dict: Results of the evaluation.
        """
        mols = []
        with torch.no_grad():
            while len(mols) < self.n_graphs:
                current = min(self.n_graphs - len(mols), self.batch_size)
                generated = pl_module.sample(current, timesteps=self.timesteps)
                generated["x"] = self.scale_coords * generated["x"]
                mols.extend(get_molecules(generated, {"atom_decoder": self.full_atom_decoder}))

        results = {}
        if return_molecules:
            results["molecules"] = deepcopy(mols)

        # Evaluate 2D stability
        mol_2d_stability = Molecule2DStability(self.dataset_info, device=pl_module.device)
        stability_res, valid_smiles, valid_molecules, stable_molecules = mol_2d_stability(
            mols)
        results.update(stability_res)

        if self.compute_2D_metrics:
            # Evaluate 2D metrics
            mol_2d_metrics = Molecule2DMetrics(self.dataset_info, device=pl_module.device)
            statistics_res = mol_2d_metrics.evaluate(valid_smiles)
            results.update(statistics_res)

        if self.compute_3D_metrics:
            # Evaluate 3D metrics
            mol_3d_metrics = Molecule3DMetrics(
                self.dataset_info, device=pl_module.device, preserve_aromatic=self.preserve_aromatic
            )
            _mols = [mol.rdkit_mol for mol in mols]
            mol_3d_res = mol_3d_metrics(_mols)
            results.update(mol_3d_res)

        if self.compute_train_data_metrics and self.train_smiles is not None:
            # Evaluate train data metrics
            train_data_metrics = MoleculeTrainDataMetrics(self.train_smiles,
                                                          device=pl_module.device)
            train_data_res = train_data_metrics(valid_smiles)
            results.update(train_data_res)
        return results

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called at the end of the validation epoch to evaluate molecules and log metrics.

        Args:
            trainer: PyTorch Lightning trainer instance.
            pl_module: PyTorch Lightning module instance.
        """
        try:
            results = self.evaluate_molecules(pl_module, trainer=trainer)
        except Exception as e:
            results = self.gather_default_values()
            print(f"The sampling has failed with the error: {e}")
        pl_module.log_dict(results, sync_dist=True)
