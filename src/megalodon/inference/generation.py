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

"""
Public API for LoQI conformer generation.

Usage:
    result = generate_conformers(
        smiles_list=["c1ccccc1", "CC(=O)O"],
        model=model,
        cfg=cfg,
        n_confs=10,          # uniform: 10 conformers per molecule
    )
    sdf = result.to_sdf()
    for smiles, mols in result.conformers.items():
        print(f"{smiles}: {len(mols)} conformers")
    for err in result.errors:
        print(f"  FAILED {err.smiles}: {err.error}")
"""

from typing import List, Optional, Union

import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader

from megalodon.inference.batching import pack_batches
from megalodon.inference.featurization import (
    build_data_list,
    debatch_conformers,
    _convert_coords_to_np,
    _write_coords_to_mol,
)
from megalodon.inference.result import ConformerGenerationResult, MoleculeProcessingError
from megalodon.inference.validation import validate_smiles


def generate_conformers(
    smiles_list: List[str],
    model: "Graph3DInterpolantModel",
    cfg: DictConfig,
    n_confs: Union[int, List[int]] = 1,
    batch_size: int = 48,
    max_atoms_per_batch: Optional[int] = None,
) -> ConformerGenerationResult:
    """
    Generate 3D conformers for a list of SMILES strings.

    Each SMILES is validated before GPU processing. Invalid SMILES are isolated
    into the `errors` field without crashing the rest of the batch.

    Args:
        smiles_list:         List of SMILES strings to process.
        model:               Loaded Graph3DInterpolantModel (already on GPU, in eval mode).
        cfg:                 OmegaConf config (needs cfg.interpolant.timesteps).
        n_confs:             Number of conformers per molecule. Either a single int
                             (same for all) or a list of ints (one per SMILES).
        batch_size:          Maximum number of graphs per DataLoader batch.
        max_atoms_per_batch: Optional int. When provided, uses FFD atom-count bin-packing
                             instead of graph-count batching. Prevents OOM for mixed molecule
                             sizes. Overrides batch_size when specified.

    Returns:
        ConformerGenerationResult with .conformers and .errors.
    """
    if not smiles_list:
        return ConformerGenerationResult(conformers={}, errors=[])

    # Normalise n_confs to a per-molecule list
    if isinstance(n_confs, int):
        n_confs_list = [n_confs] * len(smiles_list)
    else:
        if len(n_confs) != len(smiles_list):
            raise ValueError(
                f"n_confs list length ({len(n_confs)}) must match "
                f"smiles_list length ({len(smiles_list)})"
            )
        n_confs_list = list(n_confs)

    # --- Validation pass: isolate bad SMILES before touching GPU ---
    errors: List[MoleculeProcessingError] = []
    valid_entries = []
    valid_n_confs: List[int] = []

    for idx, smiles in enumerate(smiles_list):
        mol, err_msg = validate_smiles(smiles)
        if err_msg is not None:
            errors.append(MoleculeProcessingError(smiles=smiles, error=err_msg, index=idx))
        else:
            valid_entries.append((idx, smiles, mol))
            valid_n_confs.append(n_confs_list[idx])

    if not valid_entries:
        return ConformerGenerationResult(conformers={}, errors=errors)

    # --- Build PyG data list with identity tracking ---
    data_list, source_smiles_indices, _conf_indices = build_data_list(
        valid_entries, valid_n_confs
    )

    # --- GPU sampling ---
    if max_atoms_per_batch is not None:
        batches = pack_batches(data_list, max_atoms_per_batch)
    else:
        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
        batches = list(loader)

    all_generated = []

    with torch.no_grad():
        for batch in batches:
            batch = batch.to(model.device)
            sample = model.sample(
                batch=batch,
                timesteps=cfg.interpolant.timesteps,
                pre_format=True,
            )
            coords_list = _convert_coords_to_np(sample)
            mols_gen = [
                _write_coords_to_mol(mol, coords)
                for mol, coords in zip(batch["mol"], coords_list)
            ]
            all_generated.extend(mols_gen)

    # --- Reconstruct per-SMILES result ---
    conformers = debatch_conformers(all_generated, source_smiles_indices, smiles_list)

    return ConformerGenerationResult(conformers=conformers, errors=errors)
