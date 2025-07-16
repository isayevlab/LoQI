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


import math

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors
from rdkit.DataStructs import BulkTanimotoSimilarity

from megalodon.metrics.utils import get_fingerprints_from_smileslist, get_mols_list


lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

_fscores = None


def numBridgeheadsAndSpiro(mol, ri=None):
    """
    Calculate the number of bridgehead and spiro atoms in a molecule.

    Args:
        mol: RDKit molecule object.
        ri: Optional ring information.

    Returns:
        Tuple containing the number of bridgehead and spiro atoms.
    """
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def readFragmentScores(name="fpscores"):
    """
    Read fragment scores from a file.

    Args:
        name: Name of the file containing fragment scores.
    """
    import gzip
    import os.path as op
    import pickle

    global _fscores
    if name == "fpscores":
        name = op.join(op.dirname(__file__), name)
    data = pickle.load(gzip.open(f"{name}.pkl.gz"))
    _fscores = {i[j]: float(i[0]) for i in data for j in range(1, len(i))}


def calculateScore(m):
    """
    Calculate the synthetic accessibility (SA) score of a molecule.

    Args:
        m: RDKit molecule object.

    Returns:
        Synthetic accessibility score.
    """
    if _fscores is None:
        readFragmentScores()
    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)
    fps = fp.GetNonzeroElements()
    score1 = sum(_fscores.get(bitId, -4) * v for bitId, v in fps.items()) / sum(fps.values())
    sizePenalty = m.GetNumAtoms() ** 1.005 - m.GetNumAtoms()
    stereoPenalty = math.log10(len(Chem.FindMolChiralCenters(m, includeUnassigned=True)) + 1)
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = math.log10(2) if any(len(r) > 8 for r in m.GetRingInfo().AtomRings()) else 0
    score2 = 0 - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty
    score3 = math.log(m.GetNumAtoms() / len(fps)) * 0.5 if m.GetNumAtoms() > len(fps) else 0
    sascore = score1 + score2 + score3
    sascore = 11.0 - (sascore + 5) / 7.5 * 9.0
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    return min(max(sascore, 1.0), 10.0)


class Molecule2DMetrics:
    def __init__(self, dataset_info, device="cpu"):
        """
        Initialize the Molecule2DMetrics class.

        Args:
            dataset_info: Dataset information containing atom decoder.
            device: Device to use for computations.
        """
        self.atom_decoder = dataset_info["atom_decoder"]
        self.dataset_info = dataset_info
        self.device = device

    def evaluate(self, valid_smiles):
        """
        Evaluate the 2D metrics for valid SMILES strings.

        Args:
            valid_smiles: List of valid SMILES strings.

        Returns:
            Dictionary containing 2D metric values.
        """
        results = self.default_values()
        mols = get_mols_list(valid_smiles)
        if mols:
            qed_scores = [self.calculate_qed(mol) for mol in mols]
            sa_scores = [self.calculate_sa(mol) for mol in mols]
            logp_scores = [self.calculate_logp(mol) for mol in mols]
            lipinski_scores = [self.calculate_lipinski(mol) for mol in mols]

            statistics_dict = {
                "QED": np.mean(qed_scores),
                "SA": np.mean(sa_scores),
                "LogP": np.mean(logp_scores),
                "Lipinski": np.mean(lipinski_scores),
            }
            results.update(statistics_dict)
        results["Diversity"] = self.calculate_diversity(valid_smiles)
        results["Unique"] = self.calculate_unique(valid_smiles)
        return results

    def calculate_qed(self, rdmol):
        """
        Calculate the Quantitative Estimation of Drug-likeness (QED) of a molecule.

        Args:
            rdmol: RDKit molecule object.

        Returns:
            QED value.
        """
        return QED.qed(rdmol)

    def calculate_sa(self, rdmol):
        """
        Calculate the synthetic accessibility (SA) score of a molecule.

        Args:
            rdmol: RDKit molecule object.

        Returns:
            SA score.
        """
        sa = calculateScore(rdmol)
        return round((10 - sa) / 9, 2)

    def calculate_logp(self, rdmol):
        """
        Calculate the logP (octanol-water partition coefficient) of a molecule.

        Args:
            rdmol: RDKit molecule object.

        Returns:
            logP value.
        """
        return Crippen.MolLogP(rdmol)

    def calculate_lipinski(self, rdmol):
        """
        Calculate the Lipinski rule of five compliance for a molecule.

        Args:
            rdmol: RDKit molecule object.

        Returns:
            Number of Lipinski rules the molecule complies with.
        """
        rule_1 = Descriptors.ExactMolWt(rdmol) < 500
        rule_2 = Lipinski.NumHDonors(rdmol) <= 5
        rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
        logp = Crippen.MolLogP(rdmol)
        rule_4 = -2 <= logp <= 5
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
        return sum(int(rule) for rule in [rule_1, rule_2, rule_3, rule_4, rule_5])

    def calculate_diversity(self, valid_smiles):
        """
        Calculate the diversity of a set of valid SMILES strings.

        Args:
            valid_smiles: List of valid SMILES strings.

        Returns:
            Diversity value.
        """
        fps = get_fingerprints_from_smileslist(valid_smiles)
        scores = []
        for i, fp in enumerate(fps):
            fps_tmp = fps.copy()
            del fps_tmp[i]
            scores.append(BulkTanimotoSimilarity(fp, fps_tmp))
        return 1 - np.mean(scores)

    def calculate_unique(self, valid_smiles):
        """
        Calculate the uniqueness of a set of valid SMILES strings.

        Args:
            valid_smiles: List of valid SMILES strings.

        Returns:
            Uniqueness value.
        """
        unique_smiles = set(valid_smiles)
        return len(unique_smiles) / len(valid_smiles) if valid_smiles else 0.0

    @staticmethod
    def default_values():
        """
        Get default values for the 2D metrics.

        Returns:
            Dictionary of default 2D metric values.
        """
        return {"QED": 0.0, "SA": 0.0, "LogP": 0.0, "Lipinski": 0.0, "Diversity": 0.0, "Unique": 0.0}
