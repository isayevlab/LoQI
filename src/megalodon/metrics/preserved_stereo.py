from typing import Dict, List, Tuple
import torch
from rdkit import Chem

def get_stereochemistry_descriptor(mol: Chem.Mol) -> Tuple[str, str, str]:
    """
    Generate stereochemistry descriptor for a molecule.

    Parameters:
        mol (rdkit.Chem.Mol): RDKit molecule with stereo information.

    Returns:
        Tuple[str, str, str]: (R/S descriptor, Inverted R/S descriptor, E/Z descriptor)
    """
    rs_descriptor = []

    # Extract chirality (R/S) in the order they appear
    for atom in mol.GetAtoms():
        if atom.HasProp('_CIPCode'):
            rs_descriptor.append(atom.GetProp('_CIPCode'))  # 'R' or 'S'

    # Generate inverted R/S descriptor
    inv_rs_descriptor = "".join(["R" if i == "S" else "S" for i in rs_descriptor])
    rs_descriptor = "".join(rs_descriptor)

    # Extract double bond stereochemistry (E/Z) in the order they appear
    ez_descriptor = []

    for bond in mol.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREOE:
            ez_descriptor.append('E')
        elif bond.GetStereo() == Chem.BondStereo.STEREOZ:
            ez_descriptor.append('Z')

    ez_descriptor = "".join(ez_descriptor)

    return rs_descriptor, inv_rs_descriptor, ez_descriptor


class StereoMetrics:
    """
    Compute 3D stereochemistry metrics (RS Score & EZ Score) for molecules.

    The final metrics computed are:
        - `rs_score`: Fraction of molecules with correct `R/S` assignments.
        - `ez_score`: Fraction of molecules with correct `E/Z` assignments.
    """

    def __call__(self, molecules: List[Chem.Mol], reference_molecules: List[Chem.Mol]) -> Dict[
        str, float]:
        """
        Compute the stereochemistry accuracy for a set of molecules.

        Parameters:
            molecules (List[Chem.Mol]): List of predicted molecules.
            reference_molecules (List[Chem.Mol]): List of reference molecules (ground truth).

        Returns:
            Dict[str, float]: Dictionary with `rs_score` and `ez_score`.
        """
        assert len(molecules) == len(
            reference_molecules), "Molecule lists must have the same length."

        correct_rs = 0
        total_rs = 0

        correct_ez = 0
        total_ez = 0

        for mol, ref_mol in zip(molecules, reference_molecules):
            # Assign stereochemistry from 3D coordinates
            Chem.rdmolops.AssignStereochemistryFrom3D(ref_mol)
            Chem.rdmolops.AssignStereochemistryFrom3D(mol)

            # Get descriptors
            sr, inv_sr, ez = get_stereochemistry_descriptor(mol)
            ref_sr, _, ref_ez = get_stereochemistry_descriptor(ref_mol)

            # Check RS correctness
            if ref_sr:
                total_rs += 1
                if sr == ref_sr or sr == inv_sr:
                    correct_rs += 1

            # Check EZ correctness
            if ref_ez:
                total_ez += 1
                if ez == ref_ez:
                    correct_ez += 1

        # Compute final scores (handle division by zero)
        rs_score = correct_rs / total_rs if total_rs > 0 else 0.0
        ez_score = correct_ez / total_ez if total_ez > 0 else 0.0

        return {"rs_score": rs_score, "ez_score": ez_score}
