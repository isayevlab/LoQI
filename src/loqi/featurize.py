"""Molecule validation, graph construction, and conformer conversion."""

from __future__ import annotations

import os
import warnings
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from megalodon.data.adaptive_dataloader import AdaptiveBatchSampler
from megalodon.inference.validation import SUPPORTED_ELEMENTS, validate_rdkit_mol, validate_smiles
from megalodon.metrics.conformer_evaluation_callback import full_atom_encoder

CHIRAL_EDGE_TYPES = (7, 8)
EZ_EDGE_TYPES = {Chem.BondStereo.STEREOE: 5, Chem.BondStereo.STEREOZ: 6}

__all__ = [
    "CHIRAL_EDGE_TYPES",
    "EZ_EDGE_TYPES",
    "SUPPORTED_ELEMENTS",
    "add_stereo_bonds",
    "build_sampling_loader",
    "conformers_to_mol",
    "legacy_stereo_perception",
    "load_molecules",
    "mol_to_data",
    "mols_to_data_list",
    "prepare_molecule",
]


@contextmanager
def legacy_stereo_perception(enabled: bool = True) -> Iterator[None]:
    """Use the RDKit stereo perception expected by the checkpoints.

    Restore the previous setting when the context exits.
    """
    previous = Chem.GetUseLegacyStereoPerception()
    Chem.SetUseLegacyStereoPerception(enabled)
    try:
        yield
    finally:
        Chem.SetUseLegacyStereoPerception(previous)


def add_stereo_bonds(mol, chi_bonds, ez_bonds, edge_index=None, edge_attr=None, from_3D=True):
    """Add graph edges encoding tetrahedral and double-bond stereochemistry."""
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

        if bond.GetBeginAtom().HasProp("_CIPCode"):
            chirality = bond.GetBeginAtom().GetProp("_CIPCode")
            neighbors = bond.GetBeginAtom().GetNeighbors()
            if all(n.HasProp("_CIPRank") for n in neighbors):
                sorted_neighbors = sorted(neighbors, key=lambda x: int(x.GetProp("_CIPRank")), reverse=True)
                sorted_neighbors = [a.GetIdx() for a in sorted_neighbors]
                a, b, c = sorted_neighbors[:3] if chirality == "R" else sorted_neighbors[:3][::-1]
                d = sorted_neighbors[-1]
                result.extend(
                    [
                        (a, d, chi_bonds[0]),
                        (b, d, chi_bonds[0]),
                        (c, d, chi_bonds[0]),
                        (d, a, chi_bonds[0]),
                        (d, b, chi_bonds[0]),
                        (d, c, chi_bonds[0]),
                        (b, a, chi_bonds[1]),
                        (c, b, chi_bonds[1]),
                        (a, c, chi_bonds[1]),
                    ]
                )

    if not result:
        return edge_index, edge_attr
    new_edge_index = torch.tensor([[i, j] for i, j, _ in result], dtype=torch.long).T
    new_edge_attr = torch.tensor([b for _, _, b in result], dtype=torch.uint8)

    if edge_index is None:
        return new_edge_index, new_edge_attr
    edge_index = torch.cat([edge_index, new_edge_index], dim=1)
    edge_attr = torch.cat([edge_attr, new_edge_attr])
    return edge_index, edge_attr


def mol_to_data(mol: Chem.Mol, smiles: str, *, use_3d_input: bool = False, use_stereo_bonds: bool = True) -> Data:
    """Build an inference graph from an RDKit molecule with explicit hydrogens.

    Sanitize and kekulize the molecule in place. Use its first conformer's coordinates
    when ``use_3d_input=True``; otherwise initialize positions to zero.
    """
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

    if use_stereo_bonds:
        edge_index, edge_attr = add_stereo_bonds(
            mol, list(CHIRAL_EDGE_TYPES), EZ_EDGE_TYPES, edge_index, edge_attr, from_3D=use_3d_input
        )

    return Data(
        x=atom_types,
        edge_index=edge_index,
        edge_attr=edge_attr.to(torch.uint8),
        pos=pos,
        charges=all_charges,
        smiles=smiles,
        mol=mol,
        chemblid=mol.GetProp("_Name") if mol.HasProp("_Name") else "",
    )


def mols_to_data_list(
    mols: Sequence[Chem.Mol], n_confs: int = 1, *, use_3d_input: bool = False, use_stereo_bonds: bool = True
) -> list[Data]:
    """Create ``n_confs`` graph copies per molecule.

    Each copy stores its source index in ``mol_idx`` for regrouping after batching.
    """
    data_list = []
    for mol_idx, mol in enumerate(mols):
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        for _ in range(n_confs):
            copy = Chem.Mol(mol)
            data = mol_to_data(
                copy, Chem.MolToSmiles(copy), use_3d_input=use_3d_input, use_stereo_bonds=use_stereo_bonds
            )
            data.mol_idx = mol_idx
            data_list.append(data)
    return data_list


def _parse_supported_smiles(smiles: str, add_hs: bool):
    """Validate SMILES while allowing disconnected molecules with a warning."""
    mol, canonical, err = validate_smiles(smiles, add_hs=add_hs)
    if err is None:
        return mol, canonical, None, None
    if "Disconnected fragments are not supported" not in str(err):
        return None, None, err, None

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
            return (
                None,
                None,
                f"Unsupported element: '{symbol}'. Supported: {', '.join(sorted(SUPPORTED_ELEMENTS))}.",
                None,
            )
        if atom.GetNumRadicalElectrons() > 0:
            return None, None, f"Radical electrons are not supported (atom {atom.GetIdx()} {atom.GetSymbol()}).", None
    warning = "Disconnected fragments detected. Proceeding in permissive mode (experimental for dimers/multimers)."
    return mol_checked, canonical, None, warning


def prepare_molecule(smiles: str, add_hs: bool = True) -> tuple[Chem.Mol, str]:
    """Return a validated molecule and its canonical SMILES.

    Add hydrogens when requested. Unsupported elements, radicals, and invalid SMILES
    raise ``ValueError``; disconnected molecules emit ``UserWarning``.
    """
    mol, canonical, err, warning = _parse_supported_smiles(smiles, add_hs)
    if err is not None:
        raise ValueError(f"Invalid SMILES {smiles!r}: {err}")
    if warning is not None:
        warnings.warn(f"{smiles!r}: {warning}", stacklevel=2)
    return mol, canonical


def load_molecules(input_path_or_smiles: str, add_hs: bool = True) -> tuple[list[Chem.Mol], list[str]]:
    """Read a SMILES string, SMILES file, or SDF file.

    Return molecules and messages for skipped file entries. A single invalid SMILES
    string raises ``ValueError``.
    """
    errors: list[str] = []
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
            with open(input_path_or_smiles) as fh:
                smiles_list = [line.strip().split()[0] for line in fh if line.strip()]
            mols = []
            for i, smi in enumerate(smiles_list):
                mol, _, err, warning = _parse_supported_smiles(smi, add_hs=add_hs)
                if err is not None:
                    errors.append(f"SMILES line {i + 1}: {err}")
                    continue
                if warning is not None:
                    errors.append(f"SMILES line {i + 1}: WARNING: {warning}")
                mols.append(mol)
        else:
            raise ValueError(f"Unsupported input format: {input_path_or_smiles}")
    else:
        mol, _ = prepare_molecule(input_path_or_smiles, add_hs=add_hs)
        mols = [mol]
    return mols, errors


def build_sampling_loader(
    data_list: Sequence[Data],
    batch_size: int,
    *,
    atom_aware_batching: bool = True,
    shuffle: bool = False,
    target_molecule_size: int = 50,
) -> DataLoader:
    """Create a fixed-size or adaptive graph loader.

    Adaptive batches scale by ``(target_molecule_size / n_atoms)**2`` relative to
    ``batch_size`` to limit variation in the number of graph edges.
    """
    if atom_aware_batching:
        sampler = AdaptiveBatchSampler(
            data_list,
            reference_batch_size=batch_size,
            shuffle=shuffle,
            reference_size=target_molecule_size,
        )
        return DataLoader(data_list, batch_sampler=sampler)
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)


def conformers_to_mol(mol: Chem.Mol, coords_list: Sequence[np.ndarray]) -> Chem.Mol:
    """Copy a molecule and replace its conformers with the supplied coordinate arrays."""
    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    n_atoms = out.GetNumAtoms()
    for conf_id, coords in enumerate(coords_list):
        coords = np.asarray(coords, dtype=float)
        if coords.shape != (n_atoms, 3):
            raise ValueError(f"Expected coordinates of shape {(n_atoms, 3)}, got {coords.shape}.")
        conf = Chem.Conformer(n_atoms)
        conf.SetId(conf_id)
        for i in range(n_atoms):
            conf.SetAtomPosition(i, coords[i].tolist())
        out.AddConformer(conf, assignId=False)
    return out
