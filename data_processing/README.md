# Data Processing Utilities

This directory contains utilities and scripts for processing molecular datasets and converting them to PyTorch Geometric format.

> **Provenance:** `process_geom.py`, `process_qm9.py`, and their shared
> `utils_data.py` utilities were inherited from the original NVIDIA Megalodon
> codebase and retain their original license headers. `process_chembl3d.py` is
> the LoQI-specific ChEMBL3D preprocessing entry point added in this repository.

## Prerequisites

Activate the required conda environment:
```bash
conda activate megalodon
```

## Usage

### GEOM-Drugs Dataset Processing

**Step 1: Download Data**

Download MiDi **GEOM-Drugs** split files and place them in the raw folder:
```bash
mkdir -p drugs_data/raw
cd drugs_data/raw
wget -r -np -nH --cut-dirs=2 --reject "index.html*" https://bits.csb.pitt.edu/files/geom_raw/
```

**Step 2: Process Data**

```bash
# Full processing
python process_geom.py \
    --raw_data_dir /path/to/geom/raw/data \
    --save_data_folder /path/to/output/directory

# Test mode (process only 10 molecules per split)
python process_geom.py \
    --raw_data_dir ../drugs_data/raw \
    --save_data_folder ../drugs_data \
    --test_mode
```

**Input**: Directory containing `train_data.pickle`, `val_data.pickle`, `test_data.pickle`
**Output**: Processed PyTorch Geometric datasets and comprehensive statistics

### **QM9** Dataset Processing

**Data Sources**:
The **QM9** dataset can be obtained from multiple sources:
- **Primary**: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip`
- **Secondary**: `https://ndownloader.figshare.com/files/3195404`
- **Preprocessed**: `https://data.pyg.org/datasets/qm9_v3.zip`

**Download Commands**:

```bash
# Create directory for QM9 data
mkdir -p qm9_data/raw
cd qm9_data/raw

# Download main QM9 dataset
wget https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip
unzip qm9.zip
rm qm9.zip

# Download uncharacterized molecules list
wget https://ndownloader.figshare.com/files/3195404
mv 3195404 uncharacterized.txt

# Verify files are present
ls -la  # Should show: gdb9.sdf, gdb9.sdf.csv, uncharacterized.txt
```

**Processing Options**:

```bash
# Basic processing
python process_qm9.py \
    --qm9_sdf_path /path/to/gdb9.sdf \
    --save_data_folder /path/to/output/directory

# With properties and custom splits
python process_qm9.py \
    --qm9_sdf_path /path/to/gdb9.sdf \
    --save_data_folder /path/to/output/directory \
    --properties_csv /path/to/gdb9.sdf.csv \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --random_seed 42
```

**Input Files**:
- `gdb9.sdf` - Main molecular structures file
- `gdb9.sdf.csv` - Molecular properties (optional)
- `uncharacterized.txt` - List of invalid/uncharacterized molecules

**Dataset Split**:
- **Training**: 100,000 molecules (fixed)
- **Test**: 10% of total dataset  
- **Validation**: Remainder (total - 100,000 - test_size)

**Output**: Train/val/test splits with molecular properties and comprehensive statistics

## Output Structure

Processing scripts generate the following structure:
```
output_directory/
├── processed/
│   ├── train_h.pt                    # PyTorch Geometric training data
│   ├── val_h.pt                      # PyTorch Geometric validation data  
│   ├── test_h.pt                     # PyTorch Geometric test data
│   ├── train_atom_types_h.npy        # Atom type distributions
│   ├── train_bond_types_h.npy        # Bond type distributions
│   ├── train_charges_h.npy           # Charge distributions
│   ├── train_bond_lengths_h.pickle   # Bond length statistics
│   ├── train_angles_h.pickle         # Bond angle statistics
│   ├── train_dihedrals_h.pickle      # Torsion angle statistics
│   ├── train_smiles.pickle           # SMILES strings
│   └── ... (similar files for val and test)
```

## Core Utilities

- **`utils_data.py`**: Central module containing molecular geometry calculations, PyTorch Geometric conversion utilities, and statistics computation functions
- **`process_geom.py`**: Process **GEOM-Drugs** dataset
- **`process_qm9.py`**: Process **QM9** dataset

## ChEMBL3D stereo processing

`process_chembl3d.py` recreates the standard
`chembl3d_stereo/processed` training artifacts from a ChEMBL3D release. For
each `mol_id`, it selects the conformer with the lowest absolute energy across
all conformers and observed stereochemistry classes. The selected 3D geometry
is combined with the matching size-grouped SDF topology and encoded with the
same E/Z and tetrahedral stereo edges used by the inference app.

Run the small end-to-end smoke test first:

```bash
python data_processing/process_chembl3d.py \
    --dataset_dir /path/to/ChEMBL3D \
    --save_data_folder /tmp/chembl3d_smoke \
    --test_mode
```

This processes 30 molecules from group `010`, but still writes every standard
train/validation/test `.pt`, `.npy`, and `.pickle` artifact. It deliberately
does not generate custom `test_small`, `test_cremp`, or `test_rot_bonds` sets.

After the smoke test succeeds, process the complete release:

```bash
python data_processing/process_chembl3d.py \
    --dataset_dir /path/to/ChEMBL3D \
    --save_data_folder /path/to/chembl3d_stereo
```

The split is deterministic (80/10/10 with seed 42 by default). Existing
standard files are protected unless `--overwrite` is supplied. Use `--groups`
or `--limit_molecules` for other bounded runs.

## Generic 3D SDF processing

`process_sdf.py` converts any SDF containing one 3D conformer per record to
the same 42 train/validation/test artifacts used by `chembl3d_stereo`. It
preserves explicit hydrogens and applies the same E/Z and tetrahedral stereo
edge encoding as the ChEMBL3D preprocessing path.

Run a bounded test first:

```bash
python data_processing/process_sdf.py \
    --sdf_path /path/to/molecules.sdf \
    --save_data_folder /tmp/sdf_smoke \
    --test_mode
```

Then process the full SDF:

```bash
python data_processing/process_sdf.py \
    --sdf_path /path/to/molecules.sdf \
    --save_data_folder /path/to/finetuning_dataset
```

The default split is deterministic 80/10/10 with seed 42, matching the
existing `chembl3d_stereo` dataset. For a 95/2.5/2.5 split, pass
`--train_ratio 0.95 --val_ratio 0.025`. Input records must contain finite 3D
coordinates; molecule names are read from the SDF title field, with generated
IDs used when the title is empty.
