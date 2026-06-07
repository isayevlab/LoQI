from typing import Dict, List, Optional, Tuple

from rdkit import Chem


def prepare_mol_for_conformer_eval(mol: Chem.Mol, assign_from_3d: bool = False) -> Optional[Chem.Mol]:
    """Return a sanitized Kekule-form molecule for conformer-level evaluation."""
    if mol is None:
        return None

    mol = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        if assign_from_3d and mol.GetNumConformers() > 0:
            Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
        else:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    except Exception:
        return None
    return mol


def get_stereochemistry_descriptor(mol: Chem.Mol) -> Tuple[str, str, str]:
    """Generate stereochemistry descriptor for a molecule."""
    rs_descriptor = []
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            rs_descriptor.append(atom.GetProp('_CIPCode'))

    inv_rs_descriptor = ''.join(['R' if i == 'S' else 'S' for i in rs_descriptor])
    rs_descriptor = ''.join(rs_descriptor)

    ez_descriptor = []
    for bond in mol.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREOE:
            ez_descriptor.append('E')
        elif bond.GetStereo() == Chem.BondStereo.STEREOZ:
            ez_descriptor.append('Z')

    return rs_descriptor, inv_rs_descriptor, ''.join(ez_descriptor)


class StereoMetrics:
    """Compute 3D stereochemistry metrics (RS Score & EZ Score) for molecules."""

    def __call__(self, molecules: List[Chem.Mol], reference_molecules: List[Chem.Mol]) -> Dict[str, float]:
        assert len(molecules) == len(reference_molecules), 'Molecule lists must have the same length.'

        correct_rs = 0
        total_rs = 0
        correct_ez = 0
        total_ez = 0

        for mol, ref_mol in zip(molecules, reference_molecules):
            mol = prepare_mol_for_conformer_eval(mol, assign_from_3d=True)
            ref_mol = prepare_mol_for_conformer_eval(ref_mol, assign_from_3d=True)
            if mol is None or ref_mol is None:
                continue

            sr, inv_sr, ez = get_stereochemistry_descriptor(mol)
            ref_sr, _, ref_ez = get_stereochemistry_descriptor(ref_mol)

            if ref_sr:
                total_rs += 1
                if sr == ref_sr or sr == inv_sr:
                    correct_rs += 1

            if ref_ez:
                total_ez += 1
                if ez == ref_ez:
                    correct_ez += 1

        rs_score = correct_rs / total_rs if total_rs > 0 else 0.0
        ez_score = correct_ez / total_ez if total_ez > 0 else 0.0
        return {'rs_score': rs_score, 'ez_score': ez_score}
