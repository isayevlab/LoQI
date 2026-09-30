"""Stereochemistry-aware auxiliary graph edges.

LoQI uses edge classes 5/6 for E/Z relationships and 7/8 for tetrahedral
handedness. This shared module keeps dataset preprocessing and inference in
sync.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from rdkit import Chem


def is_supported_tetrahedral_center(atom: Chem.Atom) -> bool:
    """Current conditioning/evaluation scope: four explicit bonded neighbors."""
    return atom.GetDegree() == 4 and atom.GetChiralTag() in {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    }


def _append_unique(
    result: list[tuple[int, int, int]],
    seen: set[tuple[int, int, int]],
    edges: Sequence[tuple[int, int, int]],
) -> None:
    """Append non-self auxiliary edges once in construction order."""
    for edge in edges:
        if edge[0] == edge[1] or edge in seen:
            continue
        seen.add(edge)
        result.append(edge)


def _assign_stereochemistry(mol: Chem.Mol, from_3d: bool) -> None:
    if from_3d and mol.GetNumConformers() > 0:
        Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
        Chem.AssignStereochemistry(mol, cleanIt=False, force=True)
    else:
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)


def _tetrahedral_edges(
    atom: Chem.Atom,
    descriptor: str,
    canonical_ranks: Sequence[int],
    chi_bonds: Sequence[int],
    legacy_compatible: bool = False,
) -> list[tuple[int, int, int]]:
    """Encode one R/S center using its explicit ligand atoms."""
    neighbors = sorted(
        (neighbor.GetIdx() for neighbor in atom.GetNeighbors()),
        key=lambda idx: canonical_ranks[idx],
        reverse=True,
    )
    if descriptor == "S":
        if legacy_compatible and len(neighbors) == 4:
            # Original checkpoints keep the fourth ligand fixed and reverse
            # only the oriented triangle, not the entire ligand ordering.
            neighbors[:3] = reversed(neighbors[:3])
        else:
            neighbors.reverse()

    same, opposite = int(chi_bonds[0]), int(chi_bonds[1])
    if len(neighbors) == 4:
        # Preserve LoQI's original nine-edge representation, but emit it once.
        a, b, c, d = neighbors
        return [
            (a, d, same),
            (b, d, same),
            (c, d, same),
            (d, a, same),
            (d, b, same),
            (d, c, same),
            (b, a, opposite),
            (c, b, opposite),
            (a, c, opposite),
        ]

    # Other coordination geometries need a separate representation instead of
    # being silently folded into the two tetrahedral edge classes.
    return []


def add_stereo_bonds(
    mol: Chem.Mol,
    chi_bonds: Sequence[int],
    ez_bonds: Mapping[Chem.BondStereo, int],
    edge_index: torch.Tensor | None = None,
    edge_attr: torch.Tensor | None = None,
    from_3D: bool = True,
    encoding: str = "canonical",
    c_chirality: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Add E/Z and tetrahedral auxiliary edges to a molecular graph."""
    if len(chi_bonds) != 2:
        raise ValueError("chi_bonds must contain exactly two edge classes")
    if encoding not in {"canonical", "legacy_compatible"}:
        raise ValueError(f"Unknown stereo encoding: {encoding}")

    _assign_stereochemistry(mol, from_3D)
    result: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()

    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if bond.GetBondType() != Chem.BondType.DOUBLE or stereo not in ez_bonds:
            continue

        idx_3, idx_4 = bond.GetStereoAtoms()
        atom_1, atom_2 = bond.GetBeginAtom(), bond.GetEndAtom()
        idx_1, idx_2 = atom_1.GetIdx(), atom_2.GetIdx()
        idx_5 = [n.GetIdx() for n in atom_1.GetNeighbors() if n.GetIdx() not in {idx_2, idx_3}]
        idx_6 = [n.GetIdx() for n in atom_2.GetNeighbors() if n.GetIdx() not in {idx_1, idx_4}]
        inverse = Chem.BondStereo.STEREOE if stereo == Chem.BondStereo.STEREOZ else Chem.BondStereo.STEREOZ

        edges = [(idx_3, idx_4, int(ez_bonds[stereo])), (idx_4, idx_3, int(ez_bonds[stereo]))]
        if idx_5:
            edges.extend([(idx_5[0], idx_4, int(ez_bonds[inverse])), (idx_4, idx_5[0], int(ez_bonds[inverse]))])
        if idx_6:
            edges.extend([(idx_3, idx_6[0], int(ez_bonds[inverse])), (idx_6[0], idx_3, int(ez_bonds[inverse]))])
        if idx_5 and idx_6:
            edges.extend([(idx_5[0], idx_6[0], int(ez_bonds[stereo])), (idx_6[0], idx_5[0], int(ez_bonds[stereo]))])
        _append_unique(result, seen, edges)

    canonical_ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=True, includeChirality=False))
    if encoding == "legacy_compatible":
        # Preserve the priority convention used by pretrained C/P centers.
        # The canonical fallback supports centers for which RDKit omits CIP ranks.
        canonical_ranks = [int(atom.GetProp("_CIPRank")) if atom.HasProp("_CIPRank")
                           else canonical_ranks[atom.GetIdx()] for atom in mol.GetAtoms()]
    for atom in mol.GetAtoms():
        # No virtual lone-pair ligand: only four explicit neighbors are supported.
        if c_chirality and atom.GetAtomicNum() != 6:
            continue
        if not is_supported_tetrahedral_center(atom):
            continue
        if not atom.HasProp("_CIPCode"):
            continue
        descriptor = atom.GetProp("_CIPCode").upper()
        if descriptor not in {"R", "S"}:
            continue
        _append_unique(result, seen, _tetrahedral_edges(
            atom, descriptor, canonical_ranks, chi_bonds, encoding == "legacy_compatible"))

    if not result:
        return edge_index, edge_attr

    new_edge_index = torch.tensor([(start, end) for start, end, _ in result], dtype=torch.long).T
    new_edge_attr = torch.tensor([edge_type for _, _, edge_type in result], dtype=torch.uint8)
    if edge_index is None:
        return new_edge_index, new_edge_attr
    if edge_attr is None:
        raise ValueError("edge_attr is required when edge_index is provided")
    return torch.cat([edge_index, new_edge_index], dim=1), torch.cat([edge_attr, new_edge_attr])


__all__ = ["add_stereo_bonds", "is_supported_tetrahedral_center"]
