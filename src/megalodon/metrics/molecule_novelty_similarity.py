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


import logging
from typing import List, Tuple

import numpy as np
from rdkit import RDLogger
from rdkit.DataStructs import BulkTanimotoSimilarity
from torchmetrics import MeanMetric

from megalodon.metrics.utils import canonicalize_list, get_fingerprints_from_smileslist


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class MoleculeTrainDataMetrics:
    """
    Class to compute novelty and similarity metrics for generated molecules
    with respect to a training dataset.
    """

    def __init__(self, smiles_train: List[str], device: str = "cpu"):
        """
        Initialize the MoleculeTrainDataMetrics class.

        Args:
            smiles_train (List[str]): List of SMILES strings for the training dataset.
            device (str): Device to use for computations.
        """
        self.train_smiles, _ = canonicalize_list(smiles_train)
        self.train_smiles = set(self.train_smiles)
        self.train_fps = get_fingerprints_from_smileslist(self.train_smiles)
        self.similarity_metric = MeanMetric().to(device)
        self.novelty_metric = MeanMetric().to(device)

    def reset(self):
        """Reset the similarity and novelty metrics."""
        self.similarity_metric.reset()
        self.novelty_metric.reset()

    def get_bulk_similarity_with_train(self, generated_smiles: List[str]) -> float:
        """
        Compute the bulk similarity of generated SMILES with the training dataset.

        Args:
            generated_smiles (List[str]): List of generated SMILES strings.

        Returns:
            float: Average Tanimoto similarity score.
        """
        fps = get_fingerprints_from_smileslist(generated_smiles)
        scores = [BulkTanimotoSimilarity(fp, self.train_fps) for fp in fps]
        return np.mean(scores)

    def compute_novelty(self, unique: set) -> Tuple[List[str], float]:
        """
        Compute the novelty of generated SMILES with respect to the training dataset.

        Args:
            unique (set): Set of unique generated SMILES strings.

        Returns:
            List[str]: List of novel SMILES strings.
            float: Proportion of novel SMILES strings.
        """
        novel = [smiles for smiles in unique if smiles not in self.train_smiles]
        novelty_score = len(novel) / len(unique)
        return novel, novelty_score

    def __call__(self, generated_smiles: List[str]) -> dict:
        """
        Compute and return the train data metrics for the generated SMILES.

        Args:
            generated_smiles (List[str]): List of generated SMILES strings.

        Returns:
            dict: Dictionary containing novelty and similarity metrics.
        """
        unique_smiles = set(generated_smiles)
        novel, novelty_score = self.compute_novelty(unique_smiles)
        similarity_score = self.get_bulk_similarity_with_train(generated_smiles)

        self.novelty_metric(novelty_score)
        self.similarity_metric(similarity_score)

        metrics_train_data = {
            "novelty": self.novelty_metric.compute().item(),
            "similarity": self.similarity_metric.compute().item(),
        }

        return metrics_train_data

    @staticmethod
    def default_values() -> dict:
        """
        Return default values for the train data metrics.

        Returns:
            dict: Dictionary of default train data metric values.
        """
        return {
            "novelty": 0.0,
            "similarity": 0.0,
        }
