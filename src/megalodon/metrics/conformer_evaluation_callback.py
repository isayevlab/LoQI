from copy import deepcopy

import numpy as np
from lightning import pytorch as pl
from rdkit import Chem

from megalodon.metrics.molecule_metrics_3d import Molecule3DMetrics
from megalodon.metrics.molecule_metrics_aimnet2 import MoleculeAIMNet2Metrics
from megalodon.metrics.preserved_stereo import StereoMetrics

full_atom_encoder = {
    "H": 0,
    "B": 1,
    "C": 2,
    "N": 3,
    "O": 4,
    "F": 5,
    "Al": 6,
    "Si": 7,
    "P": 8,
    "S": 9,
    "Cl": 10,
    "As": 11,
    "Br": 12,
    "I": 13,
    "Hg": 14,
    "Bi": 15,
    "Se": 16
}

full_atom_decoder = dict(map(reversed, full_atom_encoder.items()))


def convert_coords_to_np(out):
    """
    Converts the output dictionary containing 'x' (coordinates) and 'batch' (molecule indices)
    into a list of NumPy arrays, where each entry represents coordinates for one molecule.

    Parameters:
        out (dict): Dictionary containing:
            - 'x' (torch.Tensor): Tensor of atomic coordinates (N, 3)
            - 'batch' (torch.Tensor): Tensor indicating molecule index for each atom

    Returns:
        List[np.ndarray]: List where each element is a NumPy array (M, 3) for a molecule.
    """
    coords_list = []

    x = out["x"].cpu().numpy()  # Convert tensor to NumPy (N, 3)
    batch = out["batch"].cpu().numpy()  # Convert tensor to NumPy (N,)

    unique_mols = np.unique(batch)  # Get unique molecule indices

    for mol_id in unique_mols:
        coords_list.append(x[batch == mol_id])  # Select coordinates for each molecule

    return coords_list


def write_coords_to_mol(mol, coord):
    """
    Embeds 3D coordinates into an RDKit molecule and assigns stereochemistry.
    """

    # Deserialize RDKit molecule
    rdkit_mol = Chem.Mol(mol)

    rdkit_mol.RemoveAllConformers()
    conf = Chem.Conformer(rdkit_mol.GetNumAtoms())

    coords = np.asarray(coord)

    for i in range(rdkit_mol.GetNumAtoms()):
        conf.SetAtomPosition(i, (float(coords[i][0]), float(coords[i][1]), float(coords[i][2])))

    rdkit_mol.AddConformer(conf)

    return rdkit_mol


class ConformerEvaluationCallback(pl.Callback):
    """
    Callback for evaluating generated molecules on various metrics.

    Args:
        n_graphs (int): Number of molecules to generate for evaluation.
        batch_size (int): Batch size for molecule generation.
        timesteps (Optional[int]): Number of timesteps for sampling.
        train_smiles (Optional[List[str]]): Training dataset SMILES strings for calculating similarity and novelty.
        statistics (Optional[Dict]): Precomputed dataset statistics.
        compute_2D_metrics (bool): Whether to compute 2D metrics.
        compute_3D_metrics (bool): Whether to compute 3D metrics.
        compute_dihedrals (bool): Whether to compute dihedral angles in 3D metrics.
        compute_train_data_metrics (bool): Whether to compute train data metrics (similarity and novelty).
    """

    def __init__(
            self,
            compute_3D_metrics=True,
            compute_energy_metrics=True,
            energy_metrics_args=None,
            compute_stereo_metrics=True,
            statistics=None,
            scale_coords=None,
            max_molecules=100,
            timesteps=100,
    ):
        super().__init__()
        self.full_atom_decoder = full_atom_decoder
        self.dataset_info = {
            "atom_decoder": full_atom_decoder,
            "atom_encoder": full_atom_encoder,
            "statistics": statistics,
        }
        self.scale_coords = scale_coords
        if scale_coords is not None:
            self.scale_coords = scale_coords
        self.molecules = []
        self.reference_molecules = []
        self.max_molecules = max_molecules
        self.timesteps = timesteps
        self.compute_3D_metrics = compute_3D_metrics
        self.compute_energy_metrics = compute_energy_metrics
        self.energy_metrics_args = energy_metrics_args
        self.compute_stereo_metrics = compute_stereo_metrics

    def gather_default_values(self):
        """
        Gather default values for metrics when evaluation fails.

        Returns:
            Dict: Default values for each metric.
        """
        defaults = {}
        if self.compute_3D_metrics:
            defaults.update(Molecule3DMetrics.default_values())
        if self.compute_energy_metrics:
            defaults.update(MoleculeAIMNet2Metrics.default_values())
        return defaults

    def evaluate_molecules(self, molecules, reference_molecules, device, return_molecules=False):
        """
        Evaluate generated molecules on specified metrics.

        Args:
            trainer: PyTorch Lightning trainer instance.
            pl_module: PyTorch Lightning module instance.

        Returns:
            Dict: Results of the evaluation.
        """
        results = {}
        if return_molecules:
            results["molecules"] = deepcopy(molecules)
        if self.compute_3D_metrics:
            mol_3d_metrics = Molecule3DMetrics(
                self.dataset_info, device=device
            )
            mol_3d_res = mol_3d_metrics(molecules)
            results.update(mol_3d_res)

        if self.compute_energy_metrics:
            energy_metrics = MoleculeAIMNet2Metrics(
                model_path=self.energy_metrics_args["model_path"],
                batchsize=self.energy_metrics_args["batchsize"],
                opt_metrics=self.energy_metrics_args["opt_metrics"],
                opt_params=self.energy_metrics_args["opt_params"],
                device=device)
            energy_res = energy_metrics(molecules, reference_molecules=reference_molecules)
            results.update(energy_res)

        if self.compute_stereo_metrics:
            streo_metrics = StereoMetrics()
            stereo_res = streo_metrics(molecules, reference_molecules)
            results.update(stereo_res)
        return results

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx,
            dataloader_idx=0):
        if len(self.molecules) < self.max_molecules:
            batch.pos = None
            out = pl_module.sample(batch=batch, timesteps=self.timesteps, pre_format=False)
            out["x"] = self.scale_coords * out["x"]
            coords_list = convert_coords_to_np(out)
            molecules = [write_coords_to_mol(mol, coord) for mol, coord in
                         zip(batch["mol"], coords_list)]
            self.molecules.extend(molecules)
            self.reference_molecules.extend(batch["mol"])

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Called at the end of the validation epoch to evaluate molecules and log metrics.

        Args:
            trainer: PyTorch Lightning trainer instance.
            pl_module: PyTorch Lightning module instance.
        """
        results = self.evaluate_molecules(self.molecules, self.reference_molecules,
                                          pl_module.device)
        self.molecules = []
        self.reference_molecules = []
        pl_module.log_dict(results, sync_dist=True)
