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
from typing import Iterable

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem


# Mute RDKit logger
RDLogger.logger().setLevel(RDLogger.CRITICAL)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def canonicalize(smiles, include_stereocenters=True, remove_hs=False):
    mol = Chem.MolFromSmiles(smiles)
    if remove_hs:
        mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=include_stereocenters) if mol else None


def canonicalize_list(smiles_list, include_stereocenters=True, remove_hs=False):
    canonicalized_smiles = [canonicalize(smiles, include_stereocenters, remove_hs) for smiles in smiles_list]
    canonicalized_smiles = [s for s in canonicalized_smiles if s]

    unique_smiles = []
    seen = set()
    duplicate_ids = []

    for i, smiles in enumerate(canonicalized_smiles):
        if smiles not in seen:
            unique_smiles.append(smiles)
            seen.add(smiles)
        else:
            duplicate_ids.append(i)

    return unique_smiles, duplicate_ids


def get_fingerprints_from_smileslist(smiles_list):
    """
    Converts the provided smiles into ECFP4 bitvectors of length 4096.

    Args:
        smiles_list: list of SMILES strings

    Returns: ECFP4 bitvectors of length 4096.

    """
    return get_fingerprints(get_mols(smiles_list))


def get_fingerprints(mols: Iterable[Chem.Mol], radius=2, length=4096):
    """
    Converts molecules to ECFP bitvectors.

    Args:
        mols: RDKit molecules
        radius: ECFP fingerprint radius
        length: number of bits

    Returns: a list of fingerprints
    """
    return [AllChem.GetMorganFingerprintAsBitVect(m, radius, length) for m in mols]


def get_mols_list(smiles_list):
    mols = []
    for i in smiles_list:
        try:
            mol = Chem.MolFromSmiles(i)
            Chem.SanitizeMol(mol)
            if mol is not None:
                mols.append(mol)
        except Exception:
            pass
    return mols


def get_mols(smiles_list: Iterable[str]) -> Iterable[Chem.Mol]:
    for i in smiles_list:
        try:
            mol = Chem.MolFromSmiles(i)
            if mol is not None:
                yield mol
        except Exception as e:
            logger.warning(e)
