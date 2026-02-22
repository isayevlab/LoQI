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

# Use legacy stereo perception to match training-time stereo assignments.
# scripts/sample_conformers.py sets this same flag at module level (line 23).
Chem.SetUseLegacyStereoPerception(True)

from megalodon.inference.batching import ffd_pack_indices, pack_batches
from megalodon.inference.generation import generate_conformers
from megalodon.inference.result import ConformerGenerationResult, MoleculeProcessingError
from megalodon.inference.validation import validate_smiles

__all__ = [
    "generate_conformers",
    "ConformerGenerationResult",
    "MoleculeProcessingError",
    "validate_smiles",
    "pack_batches",
    "ffd_pack_indices",
]
