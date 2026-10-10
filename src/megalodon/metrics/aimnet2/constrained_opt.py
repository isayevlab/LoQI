"""
Short constrained AIMNet2 relaxation, mirroring the dihedral stage of the dataset preparation
in csd_sublimation/csd_extract_nmers/quick_optimize.py (--mode dihedral, its defaults: 20 steps,
fmax 0.05 eV/A, H-X bond spring constant 10): the full geometry relaxes with one proper dihedral
per bond fixed at its current value, every heavy atom of an intermolecular contact frozen in
place, and each H-X bond restrained against over-stretching. (The preparation's hydrogen-only
stage corrected X-ray H positions in experimental structures and isn't needed here.)

Relaxing generated structures the same way the references were prepared makes their energies
comparable, unlike a full optimization of the generated structure against a constrained,
partly relaxed reference.

Contacts can't be recovered from geometry (CCDC's crystal.contacts() criterion doesn't match a
vdW-sum cutoff), so they are read from the ATOM_CONTACTS tag of the source SDF, resolved to
heavy-atom index pairs and stored as JSON keyed by SDF title (see scripts/extract_contacts.py).
The processed dataset keeps the SDF atom order and uses the title as `chemblid`.

The helpers below are copied from quick_optimize.py (with the ASE calculator swapped for the
LoQI AIMNet2 TorchScript model) so the relaxation matches the one used for the references.
"""
import json
import re

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms, FixInternals, Hookean
from ase.optimize import LBFGS
from rdkit import Chem

# Typical experimental/neutron X-H bond lengths (A); the Hookean rest length is 1.1x this.
H_BOND_MAX_LENGTHS = {
    frozenset({"C", "H"}): 1.099,
    frozenset({"N", "H"}): 1.033,
    frozenset({"O", "H"}): 1.015,
    frozenset({"S", "H"}): 1.34,
    frozenset({"P", "H"}): 1.44,
    frozenset({"Si", "H"}): 1.48,
    frozenset({"B", "H"}): 1.30,
    frozenset({"Cl", "H"}): 1.27,
    frozenset({"Br", "H"}): 1.41,
    frozenset({"F", "H"}): 0.92,
}


class AIMNet2TorchCalculator(Calculator):
    """ASE calculator around a LoQI AIMNet2 `Forces` module (energy in eV, forces in eV/A)."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, model, charge=0, device="cpu"):
        super().__init__()
        self.model = model
        self.charge = charge
        self.device = device

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        data = {
            "coord": torch.tensor(self.atoms.get_positions(), dtype=torch.float32,
                                  device=self.device).unsqueeze(0),
            "numbers": torch.tensor(self.atoms.get_atomic_numbers(), dtype=torch.long,
                                    device=self.device).unsqueeze(0),
            "charge": torch.tensor([self.charge], dtype=torch.long, device=self.device),
        }
        out = self.model(data)
        self.results["energy"] = out["energy"].detach().double().item()
        self.results["forces"] = out["forces"][0].detach().double().cpu().numpy()


class FastFixInternals(FixInternals):
    """
    FixInternals with a faster, numerically equivalent adjust_forces: the projection is
    applied as Q (Q^T f) instead of building the dense 3N x 3N projector, with the same QR
    column order as ASE.
    """

    def adjust_forces(self, atoms, forces):
        self.initialize(atoms)
        positions = atoms.positions
        N = len(forces)

        # Rigid-body translation (tx, ty, tz) and rotation (rx, ry, rz) modes
        rel = positions - positions.sum(axis=0) / N
        rigid = np.zeros((6, N, 3))
        rigid[0, :, 0] = rigid[1, :, 1] = rigid[2, :, 2] = 1.0
        rigid[3, :, 1], rigid[3, :, 2] = -rel[:, 2], rel[:, 1]
        rigid[4, :, 0], rigid[4, :, 2] = rel[:, 2], -rel[:, 0]
        rigid[5, :, 0], rigid[5, :, 1] = -rel[:, 1], rel[:, 0]
        rigid = rigid.reshape(6, -1)
        rigid /= np.linalg.norm(rigid, axis=1, keepdims=True)

        jacobians = []
        for constraint in self.constraints:
            constraint.setup_jacobian(positions)
            constraint.adjust_forces(positions, forces)
            jacobians.append(constraint.jacobian.ravel())

        # ASE inserts each constraint jacobian at the front, i.e. reversed order
        q, _ = np.linalg.qr(np.column_stack(jacobians[::-1] + list(rigid)))
        forces -= (q @ (q.T @ forces.ravel())).reshape(-1, 3)


def get_h_heavy_bonds(mol):
    """(heavy_idx, h_idx) for every hydrogen's bond to a heavy atom."""
    pairs = []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        if a.GetSymbol() == "H" and b.GetSymbol() != "H":
            pairs.append((b.GetIdx(), a.GetIdx()))
        elif b.GetSymbol() == "H" and a.GetSymbol() != "H":
            pairs.append((a.GetIdx(), b.GetIdx()))
    return pairs


def h_bond_restraints(atoms, mol, h_bond_k=10.0, h_bond_tol=0.1):
    """One-sided Hookean restraint per H-X bond (see quick_optimize.h_bond_restraints)."""
    if h_bond_k == 0:
        return []
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    restraints = []
    for heavy_idx, h_idx in get_h_heavy_bonds(mol):
        pair = frozenset({symbols[heavy_idx], "H"})
        if pair in H_BOND_MAX_LENGTHS:
            rt = H_BOND_MAX_LENGTHS[pair] * 1.1
        else:
            rt = np.linalg.norm(positions[h_idx] - positions[heavy_idx]) * (1.0 + h_bond_tol)
        restraints.append(Hookean(a1=heavy_idx, a2=h_idx, k=h_bond_k, rt=rt))
    return restraints


def find_dihedral_quadruples(positions, mol, linear_cutoff_deg=15.0):
    """One proper dihedral per bond, skipping near-linear angles (see quick_optimize)."""

    def angle_deg(a, b, c):
        v1 = positions[a] - positions[b]
        v2 = positions[c] - positions[b]
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    def pick_non_linear(candidates, b, c):
        for cand in sorted(candidates):
            angle = angle_deg(cand, b, c)
            if linear_cutoff_deg <= angle <= 180.0 - linear_cutoff_deg:
                return cand
        return None

    bonds = [tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()]
    neighbors = {i: set() for i in range(mol.GetNumAtoms())}
    for a, b in bonds:
        neighbors[a].add(b)
        neighbors[b].add(a)

    dihedrals = []
    for j, k in bonds:
        i = pick_non_linear(neighbors[j] - {k}, j, k)
        if i is None:
            continue
        l = pick_non_linear(neighbors[k] - {j, i}, k, j)
        if l is None:
            continue
        dihedrals.append([i, j, k, l])
    return dihedrals


def run_lbfgs_nan_safe(atoms, steps, fmax):
    """LBFGS that restores the starting geometry on NaN/inf (see quick_optimize).
    Returns (converged, nsteps, nan_reverted)."""
    pos_start = atoms.get_positions().copy()

    def forces_finite():
        return np.isfinite(atoms.get_forces(apply_constraint=False)).all()

    if not forces_finite():
        return False, 0, True

    opt = LBFGS(atoms, logfile=None)
    try:
        converged = opt.run(fmax=fmax, steps=steps)
    except Exception:
        if forces_finite():
            raise
        atoms.set_positions(pos_start, apply_constraint=False)
        return False, opt.nsteps, True

    if not np.isfinite(atoms.get_positions()).all():
        atoms.set_positions(pos_start, apply_constraint=False)
        return False, opt.nsteps, True
    return bool(converged), opt.nsteps, False


def constrained_relax(mol, model, contact_pairs=(), device="cpu", steps=20, fmax=0.05,
                      dihedral_epsilon=0.1, h_bond_k=10.0, h_bond_tol=0.1):
    """
    Relax `mol`'s first conformer with one proper dihedral per bond fixed, the heavy atoms of
    `contact_pairs` (0-based index pairs) frozen, and each H-X bond restrained against
    over-stretching (quick_optimize.relax_fixed_dihedrals). If the relaxation raises (e.g.
    FixInternals.adjust_positions not converging) or goes NaN, the starting geometry is kept,
    as quick_optimize.py does. steps=0 only evaluates the energy.

    Returns (positions [N, 3], energy in eV, converged, steps taken, failure message or None).
    """
    atoms = Atoms(numbers=[a.GetAtomicNum() for a in mol.GetAtoms()],
                  positions=mol.GetConformer().GetPositions())
    atoms.calc = AIMNet2TorchCalculator(model, charge=Chem.GetFormalCharge(mol), device=device)
    converged, nsteps, failure = False, 0, None

    if steps > 0:
        pos_start = atoms.get_positions().copy()
        dihedrals = [[None, quad] for quad in find_dihedral_quadruples(pos_start, mol)]
        contact_atoms = sorted({idx for pair in contact_pairs for idx in pair})
        # Hookean first so its forces are projected by FixInternals and zeroed by FixAtoms
        constraints = h_bond_restraints(atoms, mol, h_bond_k, h_bond_tol)
        constraints.append(FastFixInternals(dihedrals_deg=dihedrals, epsilon=dihedral_epsilon))
        if contact_atoms:
            constraints.append(FixAtoms(indices=contact_atoms))
        atoms.set_constraint(constraints)
        try:
            converged, nsteps, nan_reverted = run_lbfgs_nan_safe(atoms, steps, fmax)
            if nan_reverted:
                failure = "NaN forces or coordinates"
        except Exception as e:
            atoms.set_positions(pos_start, apply_constraint=False)
            converged, failure = False, str(e)
        atoms.set_constraint()

    energy = atoms.get_potential_energy()
    return atoms.get_positions(), energy, converged, nsteps, failure


# ---------------------------------------------------------------------------
# ATOM_CONTACTS parsing (format written by csd_extract_dimers.py / fix_contacts.py)
# ---------------------------------------------------------------------------

_CONTACT_LINE_RE = re.compile(
    r'^\s*\S+\s+#(\d+)\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+\.\.\.\s+'
    r'\S+\s+#(\d+)\s+[-\d.]+\s+[-\d.]+\s+[-\d.]+\s+dist\s+([-\d.]+)\s*$'
)
_HEAVY_ATOM_RE = re.compile(r'\S+ bonded to \S+\s+#(\d+)')


def parse_atom_contacts(mol_block):
    """
    Parse an SDF record's ATOM_CONTACTS field into (idx_a, idx_b, heavy_indices) tuples
    (0-based); heavy_indices gives the heavy atom bonded to each hydrogen side, in side order.
    Returns (contacts, n_unindexed), n_unindexed counting contact lines without atom indices.
    """
    contacts, n_unindexed, in_block = [], 0, False
    for line in mol_block.splitlines():
        if line.startswith('>') and '<ATOM_CONTACTS>' in line:
            in_block = True
            continue
        if not in_block:
            continue
        if not line.strip():
            break
        if line.startswith('\t'):
            if contacts:
                idx_a, idx_b, _ = contacts[-1]
                contacts[-1] = (idx_a, idx_b, [int(h) - 1 for h in _HEAVY_ATOM_RE.findall(line)])
            continue
        m = _CONTACT_LINE_RE.match(line)
        if not m:
            if '...' in line and 'dist' in line:
                n_unindexed += 1
            continue
        contacts.append((int(m.group(1)) - 1, int(m.group(2)) - 1, []))
    return contacts, n_unindexed


def resolve_contact_pairs(symbols, contacts):
    """
    Resolve parsed contacts to de-duplicated 0-based heavy-atom pairs, replacing a hydrogen
    side by its bonded heavy atom; contacts that can't be resolved are dropped
    (see quick_optimize.match_atom_contacts).
    """
    n_atoms = len(symbols)
    pairs = []
    for idx_a, idx_b, heavy in contacts:
        if not (0 <= idx_a < n_atoms and 0 <= idx_b < n_atoms):
            continue
        h_sides = [k for k, idx in enumerate((idx_a, idx_b)) if symbols[idx] == "H"]
        if len(heavy) != len(h_sides):
            continue
        resolved = [idx_a, idx_b]
        for side, heavy_idx in zip(h_sides, heavy):
            resolved[side] = heavy_idx
        i, j = resolved
        if not (0 <= i < n_atoms and 0 <= j < n_atoms):
            continue
        if symbols[i] == "H" or symbols[j] == "H" or i == j:
            continue
        pair = (min(i, j), max(i, j))
        if pair not in pairs:
            pairs.append(pair)
    return pairs


def load_contacts(path):
    """Load {molecule id: [(i, j), ...]} written by scripts/extract_contacts.py."""
    with open(path) as f:
        return {key: [tuple(p) for p in pairs] for key, pairs in json.load(f).items()}
