# Data Processing Utilities

This directory contains utilities and scripts for processing molecular datasets and converting them to PyTorch Geometric format.

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
