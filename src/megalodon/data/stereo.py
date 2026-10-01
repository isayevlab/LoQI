"""Stereochemistry-aware auxiliary graph edges.

LoQI uses edge classes 5/6 for E/Z relationships and 7/8 for tetrahedral
handedness. This shared module keeps dataset preprocessing and inference in
sync. The default uses modern, center-specific CIP ligand ordering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import warnings

import torch
from rdkit import Chem, rdBase
from rdkit.Chem import rdCIPLabeler


STEREO_ENCODING = "modern_cip"
CIP_FAILURE_PROP = '_LoQICIPFailure'


def assign_modern_cip(mol: Chem.Mol) -> None:
    """Assign bounded modern CIP labels/orders, leaving E/Z bonds unchanged.

    Never substitute legacy or canonical ranks on a labeling failure. Known
    resource-limit exceptions omit only unresolved centers, with a warning
    and a per-atom reason for callers to record. Completed centers are kept.
    Symmetric cages can require substantially more than 100k iterations.
    """
    if tuple(map(int, rdBase.rdkitVersion.split('.')[:2])) < (2026, 3):
        raise RuntimeError('Modern CIP neighbor ordering requires RDKit >= 2026.03.1')
    # Label all marked tetrahedral centers together: e.g. a three-neighbor
    # sulfur may affect a carbon ligand's CIP priority even though we do not
    # emit conditioning edges around that sulfur itself.
    centers = [a.GetIdx() for a in mol.GetAtoms() if a.GetChiralTag() in {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}]
    for atom in mol.GetAtoms():
        for prop in ('_CIPCode', '_CIPNeighborOrder', CIP_FAILURE_PROP):
            if atom.HasProp(prop):
                atom.ClearProp(prop)
    if not centers:
        return
    failure = None
    for budget in (1000000, 10000000, 1000000000):
        # Clear partial results from the previous attempt, not chiral tags.
        # All dependent centers must participate together (not one at a time).
        for atom in mol.GetAtoms():
            for prop in ('_CIPCode', '_CIPNeighborOrder'):
                if atom.HasProp(prop):
                    atom.ClearProp(prop)
        try:
            rdCIPLabeler.AssignCIPLabels(mol, atomsToLabel=centers, bondsToLabel=[], maxRecursiveIterations=budget)
            return
        except RuntimeError as exc:
            failure = str(exc)
            if 'Digraph generation failed: more than' in failure:
                break  # A larger recursion budget cannot raise this hard limit.
            if 'Max Iterations Exceeded' not in failure:
                raise
    omitted = []
    for index in centers:
        atom = mol.GetAtomWithIdx(index)
        # RDKit sets these properties only for completed primary assignments.
        # In particular, ordinary centers resolved in its fast pass are safe.
        if atom.HasProp('_CIPCode') and atom.HasProp('_CIPNeighborOrder'):
            continue
        for prop in ('_CIPCode', '_CIPNeighborOrder'):
            if atom.HasProp(prop):
                atom.ClearProp(prop)
        atom.SetProp(CIP_FAILURE_PROP, failure)
        if is_supported_tetrahedral_center(atom):
            omitted.append(index)
    if omitted:
        warnings.warn(f'Modern CIP resource limit: omitted tetrahedral edges for unresolved atom indices {omitted}; '
                      f'kept resolved centers and E/Z unchanged. {failure}', RuntimeWarning, stacklevel=2)


def _modern_cip_edges(atom: Chem.Atom, chi_bonds: Sequence[int]) -> list[tuple[int, int, int]]:
    if not atom.HasProp('_CIPCode'):
        return []  # A local tag need not denote a stereogenic center.
    props = atom.GetPropsAsDict(includePrivate=True, includeComputed=True)
    order = list(props.get('_CIPNeighborOrder', []))
    native = [n.GetIdx() for n in atom.GetNeighbors()]
    if len(order) != 4 or set(order) != set(native):
        raise ValueError(f'Missing/invalid modern CIP neighbor order at atom {atom.GetIdx()}')
    # Highest to lowest CIP priority; d remains the observer for R/S/r/s.
    permutation = [native.index(index) for index in order]
    odd = sum(permutation[i] > permutation[j] for i in range(4) for j in range(i + 1, 4)) % 2
    ccw = atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW
    if ccw != bool(odd):
        order[:3] = reversed(order[:3])
    a, b, c, d = order
    same, opposite = map(int, chi_bonds)
    return [(v, d, same) for v in (a, b, c)] + [(d, v, same) for v in (a, b, c)] + [
        (b, a, opposite), (c, b, opposite), (a, c, opposite)]


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
    if legacy_compatible and len(neighbors) == 4:
        # With distinct CIP ranks this is exactly the original R/S rule.
        # Legacy ranks can tie even when _CIPCode exists: in that case, the
        # descriptor alone cannot orient an arbitrarily tie-broken ordering.
        # Translate the local tag into this ligand order instead. Normalize
        # its signed-volume parity to the original R ordering (CW), keeping
        # the fourth, lowest-ranked ligand fixed in both configurations.
        native = [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        permutation = [native.index(index) for index in neighbors]
        odd = sum(permutation[i] > permutation[j] for i in range(4) for j in range(i + 1, 4)) % 2
        ccw = atom.GetChiralTag() == Chem.ChiralType.CHI_TETRAHEDRAL_CCW
        if ccw != bool(odd):
            neighbors[:3] = reversed(neighbors[:3])
    elif descriptor == "S":
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
    encoding: str = STEREO_ENCODING,
    c_chirality: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Add E/Z and tetrahedral auxiliary edges to a molecular graph.

    The default uses modern CIP's center-specific neighbor order, keeping its
    lowest-priority ligand as observer for R/S/r/s. Local tag parity orients
    the triangle. App, sampling and preprocessing share this convention.
    ``legacy_compatible`` retains the original legacy-CIP observer selection.

    Explicit ``encoding="canonical"`` retains the previous canonical-rank,
    full-list-reversal behavior solely for reproducing data/models made with
    that encoding. It is not compatible with the original CIP edge sets.
    """
    if len(chi_bonds) != 2:
        raise ValueError("chi_bonds must contain exactly two edge classes")
    if encoding not in {"modern_cip", "canonical", "legacy_compatible"}:
        raise ValueError(f"Unknown stereo encoding: {encoding}")

    _assign_stereochemistry(mol, from_3D)
    if encoding == "modern_cip":
        assign_modern_cip(mol)
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

    canonical_ranks = [] if encoding == "modern_cip" else list(
        Chem.CanonicalRankAtoms(mol, breakTies=True, includeChirality=False))
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
        if encoding == "modern_cip":
            _append_unique(result, seen, _modern_cip_edges(atom, chi_bonds))
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


__all__ = ["add_stereo_bonds", "assign_modern_cip", "is_supported_tetrahedral_center", "STEREO_ENCODING"]
