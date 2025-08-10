import numpy as np

covalent_radii = {
    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 10: 0.58,
    11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03,
    20: 1.76,
    21: 1.70, 22: 1.60, 23: 1.53, 24: 1.39, 25: 1.39, 26: 1.32, 27: 1.26, 28: 1.24, 29: 1.32,
    30: 1.22,
    31: 1.22, 32: 1.20, 33: 1.19, 34: 1.20, 35: 1.20, 36: 1.16, 37: 2.20, 38: 1.95, 39: 1.90,
    40: 1.75,
    41: 1.64, 42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42, 46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42,
    50: 1.39,
    51: 1.39, 52: 1.38, 53: 1.39, 54: 1.40, 55: 2.44, 56: 2.15, 57: 2.07, 58: 2.04, 59: 2.03,
    60: 2.01,
    61: 1.99, 62: 1.98, 63: 1.98, 64: 1.96
}


def get_reference_distance_matrix(adjacency_matrix, numbers):
    adjacency_mask = (adjacency_matrix > 0).astype(int)
    rads = np.array([covalent_radii[i] for i in numbers])
    return rads[:, np.newaxis] + rads[np.newaxis, :]


def calculate_distance_map(coordinates):
    diff = coordinates[:, :, np.newaxis, :] - coordinates[:, np.newaxis, :, :]
    distance_map = np.linalg.norm(diff, axis=-1)
    return distance_map


def check_topology(adjacency_matrix, numbers, coordinates, tolerance=0.4):
    """
    Check if the topology of the given molecule matches the expected covalent radii.

    Args:
        numbers (np.array): [n_atoms] Atomic numbers of the atoms in the molecule.
        adjacency_matrix (np.array): [n_atoms x n_atoms] Adjacency matrix representing bonds.
        coordinates (np.array): [k x n_atoms x 3] 3D coordinates of the atoms.
        tolerance (float): Allowed deviation from expected bond lengths.

    Returns:
        np.array: Boolean array indicating if the topology is correct for each conformer.
    """
    num_conformers = coordinates.shape[0]
    adjacency_mask = (adjacency_matrix > 0).astype(int)
    ref_dist = get_reference_distance_matrix(adjacency_matrix, numbers) * adjacency_mask
    data_dist = calculate_distance_map(coordinates) * adjacency_mask

    diffs = np.abs(data_dist - ref_dist[np.newaxis, :, :]) <= (
                ref_dist[np.newaxis, :, :] * tolerance)
    valid_topologies = diffs.all(axis=(1, 2))

    return valid_topologies

