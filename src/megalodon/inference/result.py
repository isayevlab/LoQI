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

import io
from dataclasses import dataclass, field
from typing import Dict, List

from rdkit import Chem
from rdkit.Chem import SDWriter


@dataclass
class MoleculeProcessingError:
    """Records why a SMILES string could not be processed."""
    smiles: str
    error: str
    index: int  # position in the original smiles_list


@dataclass
class ConformerGenerationResult:
    """
    Structured output from generate_conformers().

    conformers: {smiles: [RDKit Mol, ...]}  — one list per successfully processed SMILES.
                Each Mol has a 3D conformer embedded by LoQI.
    errors:     list of MoleculeProcessingError for SMILES that failed validation or sampling.
    """
    conformers: Dict[str, List]  # {smiles: [Chem.Mol]}
    errors: List[MoleculeProcessingError] = field(default_factory=list)

    @property
    def n_success(self) -> int:
        return sum(len(v) for v in self.conformers.values())

    @property
    def n_errors(self) -> int:
        return len(self.errors)

    def to_sdf(self) -> str:
        """Serialize all conformers to SDF format (string). Returns empty string if none."""
        buf = io.StringIO()
        writer = SDWriter(buf)
        for smiles, mols in self.conformers.items():
            for i, mol in enumerate(mols):
                if mol is not None and mol.GetNumConformers() > 0:
                    mol_copy = Chem.Mol(mol)
                    mol_copy.SetProp("SMILES", smiles)
                    mol_copy.SetProp("conformer_idx", str(i))
                    writer.write(mol_copy)
        writer.close()
        return buf.getvalue()
