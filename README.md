# LoQI: Scalable Low-Energy Molecular Conformer Generation with Quantum Mechanical Accuracy

<div align="center">
  <a href="https://scholar.google.com/citations?user=DOljaG8AAAAJ&hl=en" target="_blank">Filipp&nbsp;Nikitin<sup>1,2</sup></a> &emsp; <b>&middot;</b> &emsp;
  <a href="#" target="_blank">Dylan&nbsp;M.&nbsp;Anstine<sup>2,3</sup></a> &emsp; <b>&middot;</b> &emsp;
  <a href="#" target="_blank">Roman&nbsp;Zubatyuk<sup>2,5</sup></a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://scholar.google.ch/citations?user=8S0VfjoAAAAJ&hl=en" target="_blank">Saee&nbsp;Gopal&nbsp;Paliwal<sup>5</sup></a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://olexandrisayev.com/" target="_blank">Olexandr&nbsp;Isayev<sup>1,2,4*</sup></a>
  <br>
  <sup>1</sup>Ray and Stephanie Lane Computational Biology Department, Carnegie Mellon University, Pittsburgh, PA, USA
  <br>
  <sup>2</sup>Department of Chemistry, Carnegie Mellon University, Pittsburgh, PA, USA
  <br>
  <sup>3</sup>Department of Chemical Engineering and Materials Science, Michigan State University, East Lansing, MI, USA
  <br>
  <sup>4</sup>Department of Materials Science and Engineering, Carnegie Mellon University, Pittsburgh, PA, USA
  <br>
  <sup>5</sup>NVIDIA, Santa Clara, CA, USA
  <br><br>
  <a href="#" target="_blank">📄&nbsp;Paper</a> &emsp; <b>&middot;</b> &emsp;
  <a href="#citation">📖&nbsp;Citation</a> &emsp; <b>&middot;</b> &emsp;
  <a href="#setup">⚙️&nbsp;Setup</a> &emsp; <b>&middot;</b> &emsp;
  <a href="https://github.com/isayevlab/LoQI" target="_blank">🔗&nbsp;GitHub</a>
  <br><br>
  <span><sup>*</sup>Corresponding author: olexandr@olexandrisayev.com</span>
</div>

---

## Overview

<div align="center">
    <img width="700" alt="Macrocycles" src="assets/macrocycles.svg"/>
</div>

### Abstract

Molecular geometry is crucial for biological activity and chemical reactivity; however, computational methods for generating 3D structures are limited by the vast scale of conformational space and the complexities of stereochemistry. Here we present an approach that combines an expansive dataset of molecular conformers with generative diffusion models to address this problem. We introduce **ChEMBL3D**, which contains over 250 million molecular geometries for 1.8 million drug-like compounds, optimized using AIMNet2 neural network potentials to a near-quantum mechanical accuracy with implicit solvent effects included. This dataset captures complex organic molecules in various protonation states and stereochemical configurations. 

We then developed **LoQI** (Low-energy QM Informed conformer generative model), a stereochemistry-aware diffusion model that learns molecular geometry distributions directly from this data. Through graph augmentation, LoQI accurately generates molecular structures with targeted stereochemistry, representing a significant advance in modeling capabilities over previous generative methods. The model outperforms traditional approaches, achieving up to tenfold improvement in energy accuracy and effective recovery of optimal conformations. Benchmark tests on complex systems, including macrocycles and flexible molecules, as well as validation with crystal structures, show LoQI can perform low energy conformer search efficiently.

> **Note on Implementation**: LoQI is built upon the [Megalodon architecture](https://arxiv.org/pdf/2505.18392) developed, adapting it specifically for stereochemistry-aware conformer generation with the ChEMBL3D dataset.

---

## Key Features

- **ChEMBL3D Dataset**: 250+ million AIMNet2-optimized conformers for 1.8M drug-like molecules
- **Stereochemistry-Aware**: First all-atom diffusion model with explicit stereochemical encoding
- **Quantum Mechanical Accuracy**: Near-DFT accuracy with implicit solvent effects
- **Superior Performance**: Up to 10x improvement in energy accuracy over traditional methods
- **Complex Molecule Support**: Handles macrocycles, flexible molecules, and challenging stereochemistry

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended for training)
- [Conda](https://docs.conda.io/) or [Mamba](https://mamba.readthedocs.io/) (recommended)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/isayevlab/LoQI.git
cd LoQI

# Create and activate conda environment
conda create -n loqi python=3.10 -y
conda activate loqi

# Install dependencies
pip install -e .
pip install -r requirements.txt
```

### Data Setup

The training and evaluation require the **ChEMBL3D** dataset. 

**Available with this release [Download Here](https://drive.google.com/drive/folders/1PvSrep7_qIjTSslzXD3KUYEJ2Qr2lgDD?usp=sharing):**
- Pre-trained LoQI model checkpoint (`loqi.ckpt`)
- Processed ChEMBL3D lowest-energy conformers dataset (`chembl3d_stereo`)

**Coming soon:**
- Full ChEMBL3D dataset (250M+ conformers) will be released in a separate repository
- Complete dataset processing scripts and pipeline



---

## Usage

Make sure that `src` content is available in your `PYTHONPATH` (e.g., `export PYTHONPATH="./src:$PYTHONPATH"`) if LoQI is not installed locally (pip install -e .). 

### Model Training

```bash
# LoQI conformer generation model
python scripts/train.py --config-name=loqi outdir=./outputs train.gpus=1 data.dataset_root="./chembl3d_data"

# Customize training parameters
python scripts/train.py --config-name=loqi \
    outdir=./outputs \
    train.gpus=2 \
    train.n_epochs=800 \
    train.seed=42 \
    data.batch_size=150 \
    optimizer.lr=0.0001
```

### Model Inference and Sampling

#### Conformer Generation

```bash
# Generate conformers for a single molecule
python scripts/sample_conformers.py \
    --config ./conf/loqi/loqi.yaml \
    --ckpt ./data/loqi.ckpt \
    --input "c1ccccc1" \
    --output ./outputs/benzene_conformers.sdf \
    --n_confs 10 \
    --batch_size 1

# Generate conformers with evaluation
python scripts/sample_conformers.py \
    --config ./conf/loqi/loqi.yaml \
    --ckpt ./data/loqi.ckpt \
    --input "CCO" \
    --output ./outputs/ethanol_conformers.sdf \
    --n_confs 100 \
    --batch_size 10
```

#### Available Configurations

**LoQI Models:**
- `loqi.yaml` - LoQI stereochemistry-aware conformer generation model
- `nextmol.yaml` - Alternative configuration for NextMol-style generation

### Training Configuration

You can easily override configuration parameters:

```bash
# Example with custom parameters
python scripts/train.py --config-name=loqi \
    outdir=./my_training \
    run_name=my_experiment \
    train.gpus=4 \
    train.n_epochs=500 \
    data.batch_size=64 \
    data.dataset_root="/path/to/chembl3d" \
    wandb_params.mode=online
```

---

## Citation

If you use LoQI in your research, please cite our paper:

```bibtex
Citation is coming. 
```

This work builds upon the Megalodon architecture. If you use the underlying architecture, please also cite:

```bibtex
@article{reidenbach2025applications,
  title={Applications of Modular Co-Design for De Novo 3D Molecule Generation},
  author={Reidenbach, Danny and Nikitin, Filipp and Isayev, Olexandr and Paliwal, Saee},
  journal={arXiv preprint arXiv:2505.18392},
  year={2025}
}
```