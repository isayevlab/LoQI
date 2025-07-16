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

import os
import pickle
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from rdkit import Chem
import torch
import omegaconf

from megalodon.data.molecule_datamodule import MoleculeDataModule
from megalodon.metrics.molecule_evaluation_callback import MoleculeEvaluationCallback
from megalodon.models.module import Graph3DInterpolantModel
from megalodon.data.statistics import Statistics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Sample molecules using Megalodon models")
    
    # Required arguments
    parser.add_argument("--ckpt_path", type=str, required=True,
                       help="Path to the model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                       help="Path to the configuration file")
    
    # Generation settings
    parser.add_argument("--n_graphs", type=int, default=100,
                       help="Number of molecules to generate (default: 100)")
    parser.add_argument("--batch_size", type=int, default=100,
                       help="Batch size for generation (default: 100)")
    parser.add_argument("--timesteps", type=int, default=500,
                       help="Number of sampling timesteps (default: 500)")
    
    # Output settings
    parser.add_argument("--save_path", type=str, default="./sdfs",
                       help="Directory to save results (default: ./sdfs)")
    parser.add_argument("--save_name", type=str, default="generated_molecules",
                       help="Base name for saved files (default: generated_molecules)")
    parser.add_argument("--save_sdf", action="store_true",
                       help="Save molecules as SDF file")
    parser.add_argument("--save_pickle", action="store_true",
                       help="Save complete results as pickle file")
    
    # Model settings
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (default: cuda)")
    parser.add_argument("--statistics_split", type=str, default="train",
                       help="Split to use for statistics (default: train)")
    
    return parser.parse_args()


def load_model(ckpt_path, cfg):
    """
    Load the model from the checkpoint using standard PyTorch Lightning method.
    """
    print(f"Loading model from checkpoint: {ckpt_path}")
    
    # Add safe globals for loading checkpoint
    torch.serialization.add_safe_globals([omegaconf.dictconfig.DictConfig])
    
    model = Graph3DInterpolantModel.load_from_checkpoint(ckpt_path, 
                                                         interpolant_params=cfg.interpolant,
                                                         sampling_params=cfg.sample)
    print(f"✓ Model loaded successfully from {Path(ckpt_path).name}")

    # Calculate and print the number of model parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Total parameters: {total_params / 1e6:.2f}M")

    return model


def initialize_data_module(cfg, batch_size):
    """
    Initialize the data module for molecule processing.
    """
    return MoleculeDataModule(
        dataset_root=cfg.data.dataset_root,
        processed_folder="processed",
        batch_size=batch_size,
        inference_batch_size=batch_size,
        data_loader_type="midi",
    )


def initialize_eval_callback(cfg, statistics, n_graphs, batch_size, timesteps):
    """
    Initialize the evaluation callback with metrics and settings.
    """
    energy_metrics_args = OmegaConf.to_container(cfg.evaluation.energy_metrics_args, resolve=True) \
        if cfg.evaluation.energy_metrics_args is not None else None

    return MoleculeEvaluationCallback(
        n_graphs=n_graphs,
        batch_size=batch_size,
        timesteps=timesteps,
        compute_train_data_metrics=cfg.evaluation.compute_train_data_metrics,
        compute_3D_metrics=cfg.evaluation.compute_3D_metrics,
        compute_energy_metrics=cfg.evaluation.compute_energy_metrics,
        energy_metrics_args=energy_metrics_args,
        statistics=statistics,
        scale_coords=cfg.evaluation.scale_coords,
        preserve_aromatic=OmegaConf.select(cfg.evaluation, "preserve_aromatic", default=True)
    )


def save_molecules_to_sdf(molecules, save_path, save_name):
    """
    Save RDKit molecules to an SDF file.
    """
    os.makedirs(save_path, exist_ok=True)
    sdf_file_path = os.path.join(save_path, f"{save_name}.sdf")

    with open(sdf_file_path, 'w') as sdf_file:
        for mol in molecules:
            rdkit_mol = mol.raw_rdkit_mol
            mol_block = Chem.MolToMolBlock(rdkit_mol, kekulize=False)
            sdf_file.write(mol_block + "\n$$$$\n")
    
    print(f"✓ SDF file saved: {sdf_file_path}")
    return sdf_file_path


def save_results_pickle(result, save_path, save_name):
    """
    Save complete results as pickle file.
    """
    os.makedirs(save_path, exist_ok=True)
    pickle_file_path = os.path.join(save_path, f"{save_name}_results.pkl")
    
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(result, f)
    
    print(f"✓ Pickle file saved: {pickle_file_path}")
    return pickle_file_path


def print_beautiful_results(result, n_graphs, timesteps, ckpt_path):
    """
    Print the molecule generation results as simple key-value pairs.
    """
    print("\nMOLECULE GENERATION RESULTS")
    print("=" * 50)
    
    # Basic info
    print(f"target_molecules: {n_graphs}")
    if 'molecules' in result:
        print(f"generated_molecules: {len(result['molecules'])}")
    print(f"timesteps: {timesteps}")
    print(f"checkpoint: {Path(ckpt_path).name}")
    
    # Print all numerical metrics
    skip_keys = {'molecules', 'ckpt'}
    for key, value in result.items():
        if key not in skip_keys and isinstance(value, (int, float)):
            print(f"{key}: {value:.4f}")
    
    print("=" * 50)


def main():
    args = parse_args()
    
    print(f"🚀 Starting molecule generation with Megalodon")
    print(f"📁 Checkpoint: {args.ckpt_path}")
    print(f"⚙️  Config: {args.config_path}")
    print(f"🎯 Target molecules: {args.n_graphs}")
    print(f"⏱️  Timesteps: {args.timesteps}")
    print()

    # Load configuration
    cfg = OmegaConf.load(args.config_path)

    # Load model
    model = load_model(args.ckpt_path, cfg)
    model.to(args.device)
    model.eval()

    # Initialize data module and load statistics
    datamodule = initialize_data_module(cfg, args.batch_size)
    
    try:
        # Load statistics separately
        statistics = Statistics.load_statistics(
            statistics_dir=f"{cfg.data.dataset_root}/{datamodule.processed_folder}",
            split_name=args.statistics_split
        )
        print(f"✓ Statistics loaded from {args.statistics_split} split")
    except Exception as e:
        print(f"Error loading statistics: {e}")
        statistics = None
    
    eval_callback = initialize_eval_callback(cfg, statistics, args.n_graphs, args.batch_size, args.timesteps)

    # Evaluate molecules
    print(f"🧬 Generating {args.n_graphs} molecules...")
    result = eval_callback.evaluate_molecules(model, return_molecules=True)
    result['ckpt'] = args.ckpt_path

    # Print beautiful results
    print_beautiful_results(result, args.n_graphs, args.timesteps, args.ckpt_path)

    # Save results
    saved_files = []
    
    if args.save_sdf:
        sdf_path = save_molecules_to_sdf(result["molecules"], args.save_path, args.save_name)
        saved_files.append(sdf_path)

    if args.save_pickle:
        pickle_path = save_results_pickle(result, args.save_path, args.save_name)
        saved_files.append(pickle_path)

    if saved_files:
        print(f"\n💾 Files saved:")
        for file_path in saved_files:
            print(f"   • {file_path}")
    
    print(f"\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
