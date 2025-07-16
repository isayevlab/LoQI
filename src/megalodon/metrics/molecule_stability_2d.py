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
from torchmetrics import MeanMetric

from megalodon.metrics.utils import canonicalize_list


allowed_bonds = {
    "H": {0: 1, 1: 0, -1: 0},
    "C": {0: [3, 4], 1: 3, -1: 3},
    "N": {0: [2, 3], 1: [2, 3, 4], -1: 2},
    "O": {0: 2, 1: 3, -1: 1},
    "F": {0: 1, -1: 0},
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": {0: [3, 5], 1: 4},
    "S": {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
    "Cl": 1,
    "As": 3,
    "Br": {0: 1, 1: 2},
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}


def _is_valid_valence(valence, allowed, charge):
    if isinstance(allowed, int):
        valid = allowed == valence

    elif isinstance(allowed, list):
        valid = valence in allowed

    elif isinstance(allowed, dict):
        allowed = allowed.get(charge)
        if allowed is None:
            return False

        valid = _is_valid_valence(valence, allowed, charge)

    return valid


def check_stability(molecule, dataset_info, atom_decoder=None):
    """
    Check the stability of a molecule by verifying its atom types and bond types.

    Args:
        molecule: The molecule to check.
        dataset_info: Dataset information containing atom decoder.
        atom_decoder: Optional atom decoder. If not provided, it will be taken from dataset_info.

    Returns:
        Tuple of torch tensors indicating molecular stability, number of stable bonds, and the total number of bonds.
    """
    device = molecule.atom_types.device
    if atom_decoder is None:
        atom_decoder = dataset_info["atom_decoder"] if isinstance(dataset_info,
                                                                  dict) else dataset_info.atom_decoder
    atom_types = molecule.atom_types
    edge_types = torch.clone(molecule.bond_types)
    edge_types[edge_types == 4] = 1
    edge_types[edge_types < 0] = 0

    valencies = torch.sum(edge_types, dim=-1).long()

    n_stable_bonds = 0
    mol_stable = True
    for atom_type, valency, charge in zip(atom_types, valencies, molecule.charges):
        atom_type = atom_type.item()
        valency = valency.item()
        charge = charge.item()
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        is_stable = _is_valid_valence(valency, possible_bonds, charge)
        if not is_stable:
            mol_stable = False
        n_stable_bonds += int(is_stable)

    return (
        torch.tensor([mol_stable], dtype=torch.float, device=device),
        torch.tensor([n_stable_bonds], dtype=torch.float, device=device),
        len(atom_types),
    )


def is_valid_mol(rdmol):
    rdmol = Chem.Mol(rdmol)
    if rdmol is None:
        status = 0
    else:
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False)
            if len(mol_frags) > 1:
                status = 4
            else:
                largest_mol = max(mol_frags, default=rdmol, key=lambda m: m.GetNumAtoms())
                Chem.SanitizeMol(largest_mol)
                status = 5
        except Chem.rdchem.AtomValenceException:
            status = 1
        except Chem.rdchem.KekulizeException:
            status = 2
        except Chem.rdchem.AtomKekulizeException or ValueError:
            status = 3
    return status


class Molecule2DStability:
    def __init__(self, dataset_info, device="cpu"):
        """
        Initialize the Molecule2DStability class.

        Args:
            dataset_info: Dataset information containing atom decoder.
            device: Device to use for computations.
        """
        self.atom_decoder = dataset_info["atom_decoder"]
        self.dataset_info = dataset_info
        self.atom_stable = MeanMetric().to(device)
        self.mol_stable = MeanMetric().to(device)
        self.validity_metric = MeanMetric().to(device)

    def reset(self):
        """Reset the metrics."""
        for metric in [self.atom_stable, self.mol_stable, self.validity_metric]:
            metric.reset()

    def compute_validity(self, generated):
        """
        Compute the validity of the generated molecules.

        Args:
            generated: List of generated molecules.
            mode: Validation mode ("sanitized" or "raw").

        Returns:
            Tuple containing valid SMILES, valid molecules, and validity score.
        """
        valid_smiles = []
        valid_molecules = []
        
        for mol in generated:
            rdmol = mol.raw_rdkit_mol
            status = is_valid_mol(rdmol)

            if status == 5:
                rdmol = Chem.Mol(rdmol)
                smiles = Chem.MolToSmiles(rdmol)
                Chem.SanitizeMol(rdmol)
                valid_molecules.append(mol)
                valid_smiles.append(smiles)
        
        validity = len(valid_smiles) / len(generated) if generated else 0
        self.validity_metric.update(value=validity, weight=len(generated))
        
        valid_smiles, duplicate_ids = canonicalize_list(valid_smiles)
        valid_molecules = [mol for i, mol in enumerate(valid_molecules) if i not in duplicate_ids]

        return valid_smiles, valid_molecules, validity

    def evaluate(self, generated):
        """
        Evaluate the stability and validity of the generated molecules.

        Args:
            generated: List of generated molecules.

        Returns:
            Tuple containing valid SMILES, valid molecules, and validity score.
        """
        return self.compute_validity(generated)

    def __call__(self, molecules):
        """
        Evaluate the generated molecules and return their stability metrics.

        Args:
            molecules: List of generated molecules.

        Returns:
            Dictionary of stability metrics, valid SMILES, valid molecules, and stable molecules.
        """
        stable_molecules = []
        
        for mol in molecules:
            mol_stable, at_stable, num_bonds = check_stability(mol, self.dataset_info)
            self.mol_stable.update(value=mol_stable)
            self.atom_stable.update(value=at_stable / num_bonds, weight=num_bonds)

            if mol_stable:
                stable_molecules.append(mol)
        
        valid_smiles, valid_molecules, validity = self.evaluate(molecules)
        
        results = {
            "mol_stable": self.mol_stable.compute().item(),
            "atm_stable": self.atom_stable.compute().item(),
            "validity": validity,
        }
        
        return results, valid_smiles, valid_molecules, stable_molecules

    @staticmethod
    def default_values():
        """
        Get default values for the stability metrics.

        Returns:
            Dictionary of default stability metric values.
        """
        return {"mol_stable": 0, "atm_stable": 0, "validity": 0}
