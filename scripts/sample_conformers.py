import os
import pickle
from argparse import ArgumentParser
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from torch_geometric.data import DataLoader
import torch
import numpy as np
from omegaconf import OmegaConf
from copy import deepcopy
from torch_geometric.data import Data

from megalodon.models.module import Graph3DInterpolantModel
from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.data.statistics import Statistics
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


def mol_to_torch_geometric(mol, smiles, use_3d=True):
    Chem.SanitizeMol(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    adj = torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True))
    edge_index = adj.nonzero().contiguous().T
    bond_types = adj[edge_index[0], edge_index[1]]
    bond_types[bond_types == 1.5] = 4
    edge_attr = bond_types.to(torch.uint8)

    if use_3d and mol.GetNumConformers() > 0:
        pos = torch.tensor(mol.GetConformer().GetPositions()).float()
    else:
        pos = torch.zeros((mol.GetNumAtoms(), 3)).float()
        
    atom_types = torch.tensor([full_atom_encoder[atom.GetSymbol()] for atom in mol.GetAtoms()], dtype=torch.uint8)
    all_charges = torch.tensor([atom.GetFormalCharge() for atom in mol.GetAtoms()], dtype=torch.int8)

    chi_bonds = [7, 8]
    ez_bonds = {Chem.BondStereo.STEREOE: 5, Chem.BondStereo.STEREOZ: 6}
    edge_index, edge_attr = add_stereo_bonds(mol, chi_bonds, ez_bonds, edge_index, edge_attr, from_3D=use_3d)

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


def raw_to_pyg(rdkit_mol, coords=None, use_3d=True):
    if use_3d and coords is not None:
        rdkit_mol.RemoveAllConformers()
        conf = Chem.Conformer(rdkit_mol.GetNumAtoms())
        for i in range(rdkit_mol.GetNumAtoms()):
            conf.SetAtomPosition(i, tuple(coords[i]))
        rdkit_mol.AddConformer(conf)
    smiles = Chem.MolToSmiles(rdkit_mol)
    return mol_to_torch_geometric(rdkit_mol, smiles, use_3d=use_3d)


def load_rdkit_molecules(input_path_or_smiles, use_3d=True):
    if os.path.isfile(input_path_or_smiles):
        if input_path_or_smiles.endswith(".sdf"):
            suppl = Chem.SDMolSupplier(input_path_or_smiles, removeHs=False, sanitize=False)
            mols = [m for m in suppl if m is not None]
        elif input_path_or_smiles.endswith((".smi", ".smiles")):
            with open(input_path_or_smiles) as f:
                smiles_list = [line.strip().split()[0] for line in f if line.strip()]
            mols = []
            for smi in smiles_list:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    mol = Chem.AddHs(mol)
                    if use_3d:
                        try:
                            AllChem.EmbedMolecule(mol, randomSeed=42)
                            mols.append(mol)
                        except:
                            continue
                    else:
                        mols.append(mol)
        else:
            raise ValueError(f"Unsupported input format: {input_path_or_smiles}")
    else:
        # Treat it as a SMILES string
        mol = Chem.MolFromSmiles(input_path_or_smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string provided.")
        mol = Chem.AddHs(mol)
        if use_3d:
            AllChem.EmbedMolecule(mol, randomSeed=42)
        mols = [mol]
    return mols


def mols_to_data_list(mols, n_confs=1, use_3d=True):
    """Replicate each molecule n_confs times and convert to torch geometric Data objects."""
    data_list = []
    for mol in mols:
        if mol is None or mol.GetNumAtoms() == 0:
            continue
            
        if use_3d and mol.GetNumConformers() == 0:
            try:
                AllChem.EmbedMolecule(mol, randomSeed=42)
            except:
                if not use_3d:
                    pass  # Continue with zero coordinates
                else:
                    continue
                    
        pos = mol.GetConformer().GetPositions() if use_3d and mol.GetNumConformers() > 0 else None

        for _ in range(n_confs):
            data = raw_to_pyg(Chem.Mol(mol), pos, use_3d=use_3d)
            data_list.append(data)
    return data_list


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_confs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_3d", action="store_true", help="Skip 3D embedding generation")
    parser.add_argument("--skip_eval", action="store_true", help="Skip evaluation")
    args = parser.parse_args()

    # Load model
    cfg = OmegaConf.load(args.config)
    model = Graph3DInterpolantModel.load_from_checkpoint(
        args.ckpt,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preporcessor=BatchPreProcessor(cfg.data.aug_rotations, cfg.data.scale_coords)
    )
    model = model.to("cuda").eval()

    # Load molecules and replicate them n_confs times
    use_3d = not args.no_3d
    mols = load_rdkit_molecules(args.input, use_3d=use_3d)
    data_list = mols_to_data_list(mols, n_confs=args.n_confs, use_3d=use_3d)
    loader = DataLoader(data_list, batch_size=args.batch_size)

    # Sampling
    generated = []
    references = [] if not args.skip_eval else None
    ids = []
    
    for batch in tqdm(loader, desc="Sampling"):
        batch = batch.to(model.device)
        sample = model.sample(batch=batch, timesteps=cfg.interpolant.timesteps, pre_format=True)
        coords_list = convert_coords_to_np(sample)
        mols_gen = [write_coords_to_mol(mol, coords) for mol, coords in zip(batch["mol"], coords_list)]
        generated.extend(mols_gen)
        if not args.skip_eval:
            references.extend(batch["mol"])
        ids.extend([m.GetProp("_Name") if m.HasProp("_Name") else "NA" for m in batch["mol"]])

    # Save output
    if args.output.endswith(".sdf"):
        from rdkit.Chem import SDWriter
        writer = SDWriter(args.output)
        for mol in generated:
            writer.write(mol)
        writer.close()
    else:
        output_dict = {"generated": generated, "ids": ids}
        if references is not None:
            output_dict["reference"] = references
        with open(args.output, "wb") as f:
            pickle.dump(output_dict, f)

    # Evaluate only if references are available and evaluation is not skipped
    if not args.skip_eval and references:
        stats = Statistics.load_statistics(cfg.data.dataset_root + "/processed", "train")
        eval_cb = ConformerEvaluationCallback(
            timesteps=cfg.evaluation.timesteps,
            compute_3D_metrics=cfg.evaluation.compute_3D_metrics,
            compute_energy_metrics=cfg.evaluation.compute_energy_metrics,
            energy_metrics_args=OmegaConf.to_container(cfg.evaluation.energy_metrics_args,
                                                       resolve=True),
            statistics=stats,
            scale_coords=cfg.evaluation.scale_coords,
            compute_stereo_metrics=True
        )
        results = eval_cb.evaluate_molecules(generated, references, model.device)
        print("Evaluation Results:")
        print(results)

    print(f"Generated {len(generated)} conformers for {len(set(ids))} unique molecules.")


if __name__ == "__main__":
    main()