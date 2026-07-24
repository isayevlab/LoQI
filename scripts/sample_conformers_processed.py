"""
Sample conformers with a trained LoQI checkpoint and optionally evaluate them.

This is a fork of sample_conformers.py with one difference: where does the molecule
input come from.

- sample_conformers.py always builds its PyG Data objects from raw molecules
  (an .sdf/.smi/.smiles file, or a bare SMILES string) via mol_to_torch_geometric.
- This script additionally accepts a processed-dataset directory as --input
  (a data.dataset_root/processed_folder produced by the data-prep pipeline, e.g.
  .../csd_loqi_monomers_dih-relax/processed) and loads it with MoleculeDataModule,
  the same class train.py uses. --split then picks which split (train/val/test)
  to sample from. Whether --input is a file/SMILES string or a directory is
  auto-detected with os.path.isdir; everything downstream (sampling loop,
  optional AIMNet2 optimization, iRMSD pruning, output saving, evaluation) is
  shared between both modes.

Use the directory mode to sample/evaluate on an existing dataset split (e.g. to
compute test-set metrics for a checkpoint), and the file/SMILES mode to generate
conformers for arbitrary new molecules not part of any processed dataset.

Example (processed-dataset test split, with evaluation against ground-truth
conformers):
    python scripts/sample_conformers_processed.py \
        --input /data/jarret/loqi/csd_loqi_data/csd_loqi_monomers_dih-relax/processed \
        --split test \
        --config scripts/conf/loqi/loqi_finetune.yaml \
        --ckpt /data/jarret/loqi/csd_train_outputs/csd_mono_finetune/loqi_csd_mono_ft/checkpoints/last.ckpt \
        --output /tmp/test_eval.pkl \
        --eval

Example (SMILES/SDF input, no evaluation, same as sample_conformers.py):
    python scripts/sample_conformers_processed.py \
        --input molecules.sdf \
        --config scripts/conf/loqi/loqi_finetune.yaml \
        --ckpt /path/to/checkpoint.ckpt \
        --output generated.sdf
"""
import os
import pickle
from argparse import ArgumentParser, BooleanOptionalAction
from rdkit import Chem
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import torch
import numpy as np
from omegaconf import OmegaConf
from torch_geometric.data import Data

from megalodon.models.module import Graph3DInterpolantModel
from megalodon.data.adaptive_dataloader import AdaptiveBatchSampler
from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.data.molecule_dataset import MoleculeDataset
from megalodon.data.statistics import Statistics
from megalodon.inference.validation import SUPPORTED_ELEMENTS, validate_rdkit_mol, validate_smiles
from megalodon.metrics.molecule_metrics_aimnet2 import MoleculeAIMNet2Metrics
from megalodon.metrics.conformer_evaluation_callback import (
    ConformerEvaluationCallback, write_coords_to_mol, convert_coords_to_np
)

from megalodon.metrics.molecule_evaluation_callback import full_atom_encoder

Chem.SetUseLegacyStereoPerception(True)


def add_stereo_bonds(mol, chi_bonds, ez_bonds, edge_index=None, edge_attr=None, from_3D=True):
    result = []
    if from_3D and mol.GetNumConformers() > 0:
        Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
    else:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if bond.GetBondType() == Chem.BondType.DOUBLE and stereo in ez_bonds:
            idx_3, idx_4 = bond.GetStereoAtoms()
            atom_1, atom_2 = bond.GetBeginAtom(), bond.GetEndAtom()
            idx_1, idx_2 = atom_1.GetIdx(), atom_2.GetIdx()

            idx_5 = [nbr.GetIdx() for nbr in atom_1.GetNeighbors() if nbr.GetIdx() not in {idx_2, idx_3}]
            idx_6 = [nbr.GetIdx() for nbr in atom_2.GetNeighbors() if nbr.GetIdx() not in {idx_1, idx_4}]

            inv_stereo = Chem.BondStereo.STEREOE if stereo == Chem.BondStereo.STEREOZ else Chem.BondStereo.STEREOZ
            result.extend([(idx_3, idx_4, ez_bonds[stereo]), (idx_4, idx_3, ez_bonds[stereo])])

            if idx_5:
                result.extend([(idx_5[0], idx_4, ez_bonds[inv_stereo]), (idx_4, idx_5[0], ez_bonds[inv_stereo])])
            if idx_6:
                result.extend([(idx_3, idx_6[0], ez_bonds[inv_stereo]), (idx_6[0], idx_3, ez_bonds[inv_stereo])])
            if idx_5 and idx_6:
                result.extend([(idx_5[0], idx_6[0], ez_bonds[stereo]), (idx_6[0], idx_5[0], ez_bonds[stereo])])

        if bond.GetBeginAtom().HasProp('_CIPCode'):
            idx = bond.GetBeginAtom().GetIdx()
            chirality = bond.GetBeginAtom().GetProp('_CIPCode')
            neighbors = bond.GetBeginAtom().GetNeighbors()
            if all(n.HasProp("_CIPRank") for n in neighbors):
                sorted_neighbors = sorted(neighbors, key=lambda x: int(x.GetProp("_CIPRank")), reverse=True)
                sorted_neighbors = [a.GetIdx() for a in sorted_neighbors]
                a, b, c = sorted_neighbors[:3] if chirality == "R" else sorted_neighbors[:3][::-1]
                d = sorted_neighbors[-1]
                result.extend([
                    (a, d, chi_bonds[0]), (b, d, chi_bonds[0]), (c, d, chi_bonds[0]),
                    (d, a, chi_bonds[0]), (d, b, chi_bonds[0]), (d, c, chi_bonds[0]),
                    (b, a, chi_bonds[1]), (c, b, chi_bonds[1]), (a, c, chi_bonds[1])
                ])

    if not result:
        return edge_index, edge_attr
    new_edge_index = torch.tensor([[i, j] for i, j, _ in result], dtype=torch.long).T
    new_edge_attr = torch.tensor([b for _, _, b in result], dtype=torch.uint8)

    if edge_index is None:
        return new_edge_index, new_edge_attr
    edge_index = torch.cat([edge_index, new_edge_index], dim=1)
    edge_attr = torch.cat([edge_attr, new_edge_attr])
    return edge_index, edge_attr


def mol_to_torch_geometric(mol, smiles, use_3d_input=False):
    Chem.SanitizeMol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.to(torch.uint8)

    if use_3d_input and mol.GetNumConformers() > 0:
        pos = torch.tensor(mol.GetConformer().GetPositions()).float()
    else:
        pos = torch.zeros((mol.GetNumAtoms(), 3)).float()
        
    atom_types = torch.tensor([full_atom_encoder[atom.GetSymbol()] for atom in mol.GetAtoms()], dtype=torch.uint8)
    all_charges = torch.tensor([atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=torch.int8)

    chi_bonds = [7, 8]
    ez_bonds = {Chem.BondStereo.STEREOE: 5, Chem.BondStereo.STEREOZ: 6}
    edge_index, edge_attr = add_stereo_bonds(
        mol, chi_bonds, ez_bonds, edge_index, edge_attr, from_3D=use_3d_input
    )

    return Data(
        x=atom_types,
        edge_index=edge_index,
        edge_attr=edge_attr.to(torch.uint8),
        pos=pos,
        charges=all_charges,
        smiles=smiles,
        mol=mol,
        chemblid=mol.GetProp("_Name") if mol.HasProp("_Name") else ""
    )


def raw_to_pyg(rdkit_mol, use_3d_input=False):
    smiles = Chem.MolToSmiles(rdkit_mol)
    return mol_to_torch_geometric(rdkit_mol, smiles, use_3d_input=use_3d_input)


def load_rdkit_molecules(input_path_or_smiles, add_hs=True):
    def validate_smiles_allow_disconnected(smiles, add_hs=True):
        mol, canonical, err = validate_smiles(smiles, add_hs=add_hs)
        if err is None:
            return mol, canonical, None, None
        if "Disconnected fragments are not supported" not in str(err):
            return None, None, err, None

        # Permissive fallback for disconnected systems (e.g., dimers).
        mol_raw = Chem.MolFromSmiles(str(smiles).strip())
        if mol_raw is None:
            return None, None, f"RDKit failed to parse SMILES: {smiles!r}.", None
        canonical = Chem.MolToSmiles(mol_raw, canonical=True, isomericSmiles=True)
        mol_roundtrip = Chem.MolFromSmiles(canonical)
        if mol_roundtrip is None:
            return None, None, f"Revalidation failed after canonicalization: {canonical!r}.", None
        mol_checked = Chem.AddHs(mol_roundtrip) if add_hs else mol_roundtrip
        for atom in mol_checked.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in SUPPORTED_ELEMENTS:
                return None, None, (
                    f"Unsupported element: '{symbol}'. Supported: "
                    f"{', '.join(sorted(SUPPORTED_ELEMENTS))}."
                ), None
            if atom.GetNumRadicalElectrons() > 0:
                return None, None, (
                    f"Radical electrons are not supported "
                    f"(atom {atom.GetIdx()} {atom.GetSymbol()})."
                ), None
        warning = (
            "Disconnected fragments detected. Proceeding in permissive mode "
            "(experimental for dimers/multimers)."
        )
        return mol_checked, canonical, None, warning

    errors = []
    if os.path.isfile(input_path_or_smiles):
        if input_path_or_smiles.endswith(".sdf"):
            suppl = Chem.SDMolSupplier(input_path_or_smiles, removeHs=False, sanitize=False)
            mols = []
            for idx, mol in enumerate(suppl):
                if mol is None:
                    errors.append(f"SDF entry {idx}: RDKit failed to read molecule.")
                    continue
                _, err = validate_rdkit_mol(mol, add_hs=add_hs)
                if err is not None:
                    errors.append(f"SDF entry {idx}: {err}")
                    continue
                mols.append(mol)
        elif input_path_or_smiles.endswith((".smi", ".smiles")):
            with open(input_path_or_smiles) as f:
                smiles_list = [line.strip().split()[0] for line in f if line.strip()]
            mols = []
            for i, smi in enumerate(smiles_list):
                mol, _, err, warning = validate_smiles_allow_disconnected(smi, add_hs=add_hs)
                if err is not None:
                    errors.append(f"SMILES line {i + 1}: {err}")
                    continue
                if warning is not None:
                    errors.append(f"SMILES line {i + 1}: WARNING: {warning}")
                mols.append(mol)
        else:
            raise ValueError(f"Unsupported input format: {input_path_or_smiles}")
    else:
        # Treat it as a SMILES string
        mol, _, err, warning = validate_smiles_allow_disconnected(input_path_or_smiles, add_hs=add_hs)
        if err is not None:
            raise ValueError(f"Invalid SMILES string: {err}")
        if warning is not None:
            print(f"WARNING: {warning}")
        mols = [mol]
    return mols, errors


def mols_to_data_list(mols, n_confs=1, use_3d_input=False):
    """Replicate each molecule n_confs times and convert to torch geometric Data objects."""
    data_list = []
    for mol in mols:
        if mol is None or mol.GetNumAtoms() == 0:
            continue
            
        for _ in range(n_confs):
            data = raw_to_pyg(Chem.Mol(mol), use_3d_input=use_3d_input)
            data_list.append(data)
    return data_list


def build_sampling_loader(
        data_list,
        sample_batch_size,
        atom_aware_batching=True,
        shuffle=False,
        target_molecule_size=50,
):
    if atom_aware_batching:
        sampler = AdaptiveBatchSampler(
            data_list,
            reference_batch_size=sample_batch_size,
            shuffle=shuffle,
            reference_size=target_molecule_size,
        )
        return DataLoader(data_list, batch_sampler=sampler)
    return DataLoader(data_list, batch_size=sample_batch_size, shuffle=shuffle)


def optimize_with_aimnet(
        molecules,
        cfg,
        opt_batch_size=None,
        fmax=0.05,
        max_nstep=250,
):
    aimnet_path = cfg.evaluation.energy_metrics_args.model_path
    if not os.path.exists(str(aimnet_path)):
        return None, None, f"AIMNet2 model not found: {aimnet_path}"
    if opt_batch_size is None:
        opt_batch_size = int(getattr(cfg.evaluation.energy_metrics_args, "batchsize", 100))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    energy_metrics = MoleculeAIMNet2Metrics(
        model_path=str(aimnet_path),
        batchsize=int(opt_batch_size),
        opt_metrics=True,
        opt_params={"fmax": float(fmax), "max_nstep": int(max_nstep)},
        device=device,
    )
    try:
        _, _, opt_mols, opt_energies = energy_metrics(
            molecules, reference_molecules=None, return_molecules=True
        )
        return opt_mols, opt_energies, None
    except Exception as exc:
        return None, None, f"Optimization failed: {exc}"


def select_unique_with_irmsd(molecules, rthr=0.125):
    if molecules is None:
        return None, None, "No molecules provided for iRMSD pruning."
    if len(molecules) == 0:
        return [], [], None
    if len(molecules) == 1:
        # Nothing to prune for a single conformer.
        return molecules, [0], None

    try:
        from irmsd import sorter_irmsd_rdkit  # type: ignore
    except Exception:
        return None, None, "iRMSD is not installed. Install with: pip install irmsd"
    try:
        # iinversion=2 disables inversion.
        groups, _ = sorter_irmsd_rdkit(
            molecules, rthr=float(rthr), iinversion=2, allcanon=True, printlvl=0
        )
        groups = np.asarray(groups).reshape(-1)
        if groups.shape[0] != len(molecules):
            return None, None, (
                f"iRMSD returned unexpected group shape {groups.shape}; expected ({len(molecules)},)."
            )
        selected_indices = []
        seen = set()
        for idx, gid in enumerate(groups.tolist()):
            if gid not in seen:
                seen.add(gid)
                selected_indices.append(idx)
        if not selected_indices:
            return None, None, "iRMSD did not produce any unique representatives."
        unique_mols = [molecules[i] for i in selected_indices]
        return unique_mols, selected_indices, None
    except Exception as exc:
        return None, None, f"iRMSD pruning failed: {exc}"


def select_subset_indices(n, max_molecules, seed):
    """Deterministically pick `max_molecules` indices out of range(n), reproducible via `seed`.

    Used to compare checkpoints (e.g. finetuned vs. pretrained) on the identical subset.
    """
    if max_molecules is None or max_molecules >= n:
        return list(range(n))
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(n, size=max_molecules, replace=False).tolist())


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help=(
            "Either a .sdf/.smi/.smiles file (or a raw SMILES string), sampled via the "
            "molecule-conversion path below, OR a processed-dataset directory "
            "(e.g. .../processed), in which case --split selects train/val/test."
        ),
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="When --input is a processed-dataset directory, which split to load.",
    )
    parser.add_argument("--n_confs", type=int, default=1)
    parser.add_argument(
        "--max_molecules",
        type=int,
        default=None,
        help=(
            "If set, sample only this many molecules from the split/input (before --n_confs "
            "replication), chosen reproducibly via --subset_seed -- e.g. to compare a "
            "finetuned vs. pretrained checkpoint on the identical subset."
        ),
    )
    parser.add_argument(
        "--subset_seed",
        type=int,
        default=42,
        help="Random seed used to select --max_molecules molecules reproducibly.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Sampling batch size. If omitted, uses data.inference_batch_size from config.",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=25,
        help=(
            "Sampling steps (default: 25). Diffusion models were trained with 25 steps and "
            "are not expected to work well for other values. Flow-matching models can be "
            "run with different step counts."
        ),
    )
    parser.add_argument(
        "--add-hs",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "Add hydrogens during SMILES validation/featurization. "
            "Use --no-add-hs when input already contains explicit hydrogens."
        ),
    )
    parser.add_argument("--eval", action="store_true", help="Run evaluation (off by default)")
    parser.add_argument(
        "--postprocess",
        choices=["none", "optimization", "optimization+irmsd"],
        default="none",
        help="Optional postprocessing of generated conformers.",
    )
    parser.add_argument(
        "--optimization_batch_size",
        type=int,
        default=None,
        help="Batch size for AIMNet2 optimization (default: cfg.evaluation.energy_metrics_args.batchsize).",
    )
    parser.add_argument(
        "--opt_fmax",
        type=float,
        default=None,
        help="Optimization force threshold (default: cfg.evaluation.energy_metrics_args.opt_params.fmax).",
    )
    parser.add_argument(
        "--opt_max_nstep",
        type=int,
        default=None,
        help="Maximum optimization steps (default: cfg.evaluation.energy_metrics_args.opt_params.max_nstep).",
    )
    parser.add_argument("--irmsd_rthr", type=float, default=0.125, help="iRMSD pruning threshold.")
    parser.add_argument(
        "--atom-aware-batching",
        action=BooleanOptionalAction,
        default=True,
        help=(
            "Enable atom-aware batching with AdaptiveBatchSampler. "
            "Use --no-atom-aware-batching to disable."
        ),
    )
    parser.add_argument(
        "--target-molecule-size",
        type=int,
        default=50,
        help="Target molecule size for atom-aware batching (default: 50).",
    )
    parser.add_argument(
        "--shuffle",
        action=BooleanOptionalAction,
        default=False,
        help="Shuffle conformer replicas before batching. Use --no-shuffle to disable.",
    )
    args = parser.parse_args()

    # Load model
    cfg = OmegaConf.load(args.config)
    cfg_opt_params = getattr(getattr(cfg.evaluation, "energy_metrics_args", None), "opt_params", None)
    opt_fmax = (
        float(args.opt_fmax)
        if args.opt_fmax is not None
        else float(getattr(cfg_opt_params, "fmax", 0.05))
    )
    opt_max_nstep = (
        int(args.opt_max_nstep)
        if args.opt_max_nstep is not None
        else int(getattr(cfg_opt_params, "max_nstep", 250))
    )
    sample_batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(getattr(cfg.data, "inference_batch_size", getattr(cfg.data, "batch_size", 32)))
    )
    atom_aware_batching = bool(args.atom_aware_batching)
    shuffle = bool(args.shuffle)
    target_molecule_size = int(args.target_molecule_size)
    model = Graph3DInterpolantModel.load_from_checkpoint(
        args.ckpt,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=BatchPreProcessor(cfg.data.aug_rotations, cfg.data.scale_coords)
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    processed_stats_dir = f"{cfg.data.dataset_root}/processed"

    if os.path.isdir(args.input):
        # Processed-dataset path: load the requested split directly via MoleculeDataset (the
        # same on-disk Data objects train.py uses), replicate each molecule --n_confs times,
        # and sample through build_sampling_loader/AdaptiveBatchSampler -- the same path used
        # for SMILES/SDF input below. AdaptiveBatchSampler tracks skipped indices and requeues
        # them, guaranteeing every (molecule, replica) pair is enumerated exactly once; the
        # "midi" loader used by train.py for train/val has no such guarantee (it silently
        # drops molecules that don't fit its size-bucket window, tolerable there only because
        # shuffling across many epochs eventually covers most of the data).
        dataset_root = os.path.dirname(os.path.normpath(args.input))
        processed_folder = os.path.basename(os.path.normpath(args.input))
        split_dataset = MoleculeDataset(root=dataset_root, processed_folder=processed_folder, split=args.split)
        subset_indices = select_subset_indices(len(split_dataset), args.max_molecules, args.subset_seed)
        data_list = []
        for idx in subset_indices:
            data = split_dataset[idx]
            for _ in range(args.n_confs):
                replica = data.clone()
                replica.mol = Chem.Mol(data.mol)
                data_list.append(replica)
        print(f"Loaded {len(data_list)} conformer replicas from {len(subset_indices)} molecules in split '{args.split}' of processed dataset '{processed_folder}'.")
        loader = build_sampling_loader(
            data_list=data_list,
            sample_batch_size=sample_batch_size,
            atom_aware_batching=atom_aware_batching,
            shuffle=shuffle,
            target_molecule_size=target_molecule_size,
        )
        print(f"Sampling from processed dataset '{processed_folder}' split '{args.split}' with {len(loader)} batches of size {sample_batch_size}.")
        has_3d_input = True  # processed dataset molecules always carry a reference conformer
        processed_stats_dir = f"{dataset_root}/{processed_folder}"
    else:
        # SMILES/SDF path: load molecules and replicate them n_confs times.
        # Use provided 3D coordinates only for SDF inputs that already contain conformers.
        input_is_sdf = os.path.isfile(args.input) and args.input.endswith(".sdf")
        mols, validation_errors = load_rdkit_molecules(args.input, add_hs=args.add_hs)
        for err in validation_errors:
            print(f"WARNING: {err}")
        if not mols:
            raise ValueError("No valid molecules left after validation/revalidation checks.")
        subset_indices = select_subset_indices(len(mols), args.max_molecules, args.subset_seed)
        mols = [mols[i] for i in subset_indices]
        has_3d_input = any(mol.GetNumConformers() > 0 for mol in mols) if input_is_sdf else False
        use_3d_input = input_is_sdf and has_3d_input
        data_list = mols_to_data_list(mols, n_confs=args.n_confs, use_3d_input=use_3d_input)
        loader = build_sampling_loader(
            data_list=data_list,
            sample_batch_size=sample_batch_size,
            atom_aware_batching=atom_aware_batching,
            shuffle=shuffle,
            target_molecule_size=target_molecule_size,
        )

    # Sampling
    generated = []
    skip_eval = not args.eval
    references = [] if not skip_eval else None
    ids = []
    timesteps = args.n_steps
    
    for batch in tqdm(loader, desc="Sampling"):
        batch = batch.to(model.device)
        sample = model.sample(batch=batch, timesteps=timesteps, pre_format=True)
        coords_list = convert_coords_to_np(sample)
        mols_gen = [write_coords_to_mol(mol, coords) for mol, coords in zip(batch["mol"], coords_list)]
        generated.extend(mols_gen)
        if not skip_eval:
            references.extend(batch["mol"])
        ids.extend([m.GetProp("_Name") if m.HasProp("_Name") else "NA" for m in batch["mol"]])

    energies = None
    if args.postprocess in {"optimization", "optimization+irmsd"}:
        optimized, energies, opt_error = optimize_with_aimnet(
            generated,
            cfg,
            opt_batch_size=args.optimization_batch_size,
            fmax=opt_fmax,
            max_nstep=opt_max_nstep,
        )
        if opt_error is not None:
            raise RuntimeError(opt_error)
        generated = optimized
        if hasattr(energies, "detach"):
            energies = energies.detach().cpu().numpy()
        else:
            energies = np.asarray(energies)
        print(f"Optimization complete: {len(generated)} conformers.")

    if args.postprocess == "optimization+irmsd":
        unique_mols, selected_indices, irmsd_error = select_unique_with_irmsd(
            generated, rthr=args.irmsd_rthr
        )
        if irmsd_error is not None:
            raise RuntimeError(irmsd_error)
        generated = unique_mols
        ids = [ids[i] for i in selected_indices]
        if references is not None:
            references = [references[i] for i in selected_indices]
        if energies is not None:
            energies = energies[selected_indices]
        print(f"iRMSD unique selection complete: {len(generated)} conformers.")

    # Save output
    if args.output.endswith(".sdf"):
        from rdkit.Chem import SDWriter
        writer = SDWriter(args.output)
        ev2kcalpermol = 23.060547830619026
        for idx, mol in enumerate(generated):
            if energies is not None:
                mol.SetProp("Energy_kcal_mol", f"{float(energies[idx]) * ev2kcalpermol:.6f}")
            writer.write(mol)
        writer.close()
    else:
        output_dict = {"generated": generated, "ids": ids}
        if energies is not None:
            output_dict["energies"] = energies
        if references is not None:
            output_dict["reference"] = references
        with open(args.output, "wb") as f:
            pickle.dump(output_dict, f)

    # Evaluate only if references are available and evaluation is not skipped
    if not skip_eval and references and has_3d_input:
        stats = Statistics.load_statistics(processed_stats_dir, "train")
        eval_cb = ConformerEvaluationCallback(
            timesteps=timesteps,
            compute_3D_metrics=cfg.evaluation.compute_3D_metrics,
            compute_energy_metrics=cfg.evaluation.compute_energy_metrics,
            energy_metrics_args=OmegaConf.to_container(cfg.evaluation.energy_metrics_args,
                                                       resolve=True),
            statistics=stats,
            scale_coords=cfg.evaluation.scale_coords,
            compute_stereo_metrics=True
        )
        for gen, ref in zip(generated, references):
            if ref.GetNumConformers() == 0:
                ref.AddConformer(Chem.Conformer(ref.GetNumAtoms()))
                conf = gen.GetConformer(0)
                pos = conf.GetPositions()
                conf.SetPositions(pos)
                ref.AddConformer(conf)
        results = eval_cb.evaluate_molecules(generated, reference_molecules=references, device=model.device)
        print("Evaluation Results:")
        print(results)

    print(f"Generated {len(generated)} conformers for {len(set(ids))} unique molecules.")


if __name__ == "__main__":
    main()
