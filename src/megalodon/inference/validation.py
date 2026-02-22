# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

from rdkit import Chem

SUPPORTED_ELEMENTS = {
    "H", "B", "C", "N", "O", "F", "Al", "Si",
    "P", "S", "Cl", "As", "Br", "I", "Hg", "Bi", "Se"
}


def validate_smiles(smiles: str) -> Tuple[Optional[Chem.Mol], Optional[str]]:
    """
    Validate a SMILES string for LoQI compatibility.

    Returns:
        (mol, None)   — RDKit Mol with Hs added, ready for featurization
        (None, error) — string describing why the SMILES is invalid
    """
    if not smiles or not smiles.strip():
        return None, "Empty SMILES string"

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, f"RDKit failed to parse SMILES: {smiles!r}"

    # Disconnected fragments (e.g. salts, mixtures) are not supported
    if len(Chem.GetMolFrags(mol)) > 1:
        return None, "Disconnected fragments (e.g. salts) are not supported"

    # Add hydrogens before element check — some implicit Hs become explicit
    mol_h = Chem.AddHs(mol)

    for atom in mol_h.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in SUPPORTED_ELEMENTS:
            return None, f"Unsupported element: {sym!r} (not in LoQI atom vocabulary)"
        if atom.GetNumRadicalElectrons() > 0:
            return None, (
                f"Radical electrons on {atom.GetSymbol()} (index {atom.GetIdx()}) "
                f"are not supported"
            )

    return mol_h, None
