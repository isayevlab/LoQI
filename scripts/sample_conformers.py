import os
from argparse import ArgumentParser
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import DataLoader
import torch
import numpy as np
from omegaconf import OmegaConf
from copy import copy, deepcopy
from torch_geometric.data import Data

from megalodon.models.module import Graph3DInterpolantModel
from megalodon.data.batch_preprocessor import BatchPreProcessor

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

        # Build topology once, then replicate cheaply
        base_data = raw_to_pyg(Chem.Mol(mol), pos, use_3d=use_3d)
        for _ in range(n_confs):
            data_list.append(copy(base_data))
    return data_list


def main():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n_confs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--no_3d", action="store_true", help="Skip 3D embedding for SDF input preprocessing (does not affect diffusion-based coordinate generation)")
    parser.add_argument("--skip_eval", action="store_true", help="(Ignored) Evaluation metrics are not computed by this script; kept for backward compatibility")
    args = parser.parse_args()

    # Load model
    cfg = OmegaConf.load(args.config)
    model = Graph3DInterpolantModel.load_from_checkpoint(
        args.ckpt,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=BatchPreProcessor(cfg.data.aug_rotations, cfg.data.scale_coords)
    )
    model = model.to("cuda").eval()

    # Load molecules
    use_3d = not args.no_3d
    mols = load_rdkit_molecules(args.input, use_3d=use_3d)

    # Build SMILES list and preserve _Name property from SDF inputs.
    # Chem.MolToSmiles() produces canonical SMILES, which becomes the key in
    # ConformerGenerationResult.conformers. Map canonical SMILES back to the
    # original mol name so pickle output stays compatible with the old format.
    smiles_list = [Chem.MolToSmiles(m) for m in mols]
    smiles_to_name = {
        smi: (m.GetProp("_Name") if m.HasProp("_Name") else smi)
        for smi, m in zip(smiles_list, mols)
    }

    # Sampling via inference API
    from megalodon.inference import generate_conformers
    result = generate_conformers(
        smiles_list=smiles_list,
        model=model,
        cfg=cfg,
        n_confs=args.n_confs,
        batch_size=args.batch_size,
    )

    generated = []
    ids = []
    for smiles, conf_mols in result.conformers.items():
        generated.extend(conf_mols)
        name = smiles_to_name.get(smiles, smiles)
        ids.extend([name] * len(conf_mols))

    for err in result.errors:
        print(f"WARNING: skipped SMILES at index {err.index}: {err.error}")

    # Save output
    if args.output.endswith(".sdf"):
        with open(args.output, "w") as f:
            f.write(result.to_sdf())
    else:
        import pickle
        output_dict = {"generated": generated, "ids": ids}
        with open(args.output, "wb") as f:
            pickle.dump(output_dict, f)

    print(f"Generated {result.n_success} conformers for "
          f"{len(result.conformers)} unique molecules "
          f"({result.n_errors} SMILES failed validation).")


if __name__ == "__main__":
    main()