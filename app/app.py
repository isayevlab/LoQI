import streamlit as st
import sys
import os
import numpy as np
import torch
from stmol import showmol
from rdkit import Chem
from rdkit.Chem import Draw
import py3Dmol
from omegaconf import OmegaConf
import pandas as pd

# Add src to path for imports
sys.path.append('/home/fnikitin/LoQI/src')

from megalodon.models.module import Graph3DInterpolantModel
from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.metrics.molecule_metrics_aimnet2 import MoleculeAIMNet2Metrics

# Import utility functions
from utils import (
    generate_conformers_batch, 
    create_sdf_content, 
    safe_filename_from_smiles,
    get_energy_statistics,
    check_topology_preservation,
    check_stereochemistry_preservation
)

# Set legacy stereo perception
Chem.SetUseLegacyStereoPerception(True)

# Streamlit app configuration
st.set_page_config(
    page_title="LoQI Conformer Generator",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 LoQI: Low-Energy QM Informed Conformer Generator")
st.markdown("Generate and visualize low-energy molecular conformers with quantum mechanical accuracy")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Add a button to clear cache if needed
if st.sidebar.button("Clear Model Cache"):
    st.cache_resource.clear()
    st.sidebar.success("Cache cleared! The app will reload the model on next generation.")

# Load model function (cached)
@st.cache_resource
def load_model():
    """Load the LoQI model and configuration"""
    config_path = "/home/fnikitin/LoQI/scripts/conf/loqi/loqi.yaml"
    ckpt_path = "/home/fnikitin/LoQI/data/loqi.ckpt"
    
    cfg = OmegaConf.load(config_path)
    
    # Update the dataset_root to the correct path
    cfg.data.dataset_root = "/home/fnikitin/LoQI/data/chembl3d_stereo"
    
    model = Graph3DInterpolantModel.load_from_checkpoint(
        ckpt_path,
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preporcessor=BatchPreProcessor(cfg.data.aug_rotations, cfg.data.scale_coords)
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    
    return model, cfg

# Utility functions are now imported from utils.py

def evaluate_energies(molecules, cfg):
    """Evaluate energies using AIMNet2"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if AIMNet2 model exists
    aimnet_path = "/home/fnikitin/LoQI/src/megalodon/metrics/aimnet2/cpcm_model/wb97m_cpcms_v2_0.jpt"
    if not os.path.exists(aimnet_path):
        return None, None, "AIMNet2 model not found"
    
    energy_metrics = MoleculeAIMNet2Metrics(
        model_path=aimnet_path,
        batchsize=100,
        opt_metrics=True,
        opt_params={
            "fmax": 2e-3,
            "max_nstep": 3000
        },
        device=device
    )
    
    # Evaluate with optimization
    results, valid_mols, opt_mols, opt_energies = energy_metrics(
        molecules, 
        reference_molecules=None, 
        return_molecules=True
    )
    
    return opt_mols, opt_energies, results

# Main app interface
col1, col2 = st.columns([1, 2])

with col1:
    st.header("Input")
    
    # SMILES input
    smiles = st.text_input("Enter SMILES", "CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/CO)/C)/C", help="Enter a valid SMILES string")
    
    # Number of conformers
    n_confs = st.slider("Number of conformers", min_value=1, max_value=20, value=10)
    
    # Generate button
    generate_button = st.button("Generate Conformers", type="primary")

with col2:
    st.header("Molecule Visualization")
    
    if smiles:
        # Show 2D structure
        try:
            mol_2d = Chem.MolFromSmiles(smiles)
            if mol_2d:
                img = Draw.MolToImage(mol_2d, size=(400, 300))
                st.image(img, caption="2D Structure")
            else:
                st.error("Invalid SMILES string")
        except Exception as e:
            st.error(f"Error drawing 2D structure: {str(e)}")

# Results section
if generate_button and smiles:
    with st.spinner("Loading LoQI model..."):
        model, cfg = load_model()
    
    with st.spinner(f"Generating {n_confs} conformers..."):
        generated_mols, reference_mols, error = generate_conformers_batch(smiles, model, cfg, n_confs)
    
    if error:
        st.error(f"Error generating conformers: {error}")
    elif generated_mols:
        st.success(f"Generated {len(generated_mols)} conformers successfully!")
        
        with st.spinner("Evaluating energies with AIMNet2..."):
            opt_mols, opt_energies, results = evaluate_energies(generated_mols, cfg)
        
        # Check topology and stereochemistry preservation
        with st.spinner("Checking topology and stereochemistry preservation..."):
            topology_results = check_topology_preservation(opt_mols)
            stereo_results = check_stereochemistry_preservation( opt_mols, reference_mols)
        
        ev2kcalpermol = 23.060547830619026
        energies_kcal = opt_energies.cpu().numpy() * ev2kcalpermol
        
        # Get energy statistics with preservation info
        energy_stats = get_energy_statistics(energies_kcal, topology_results, stereo_results)
        
        # Determine which conformer to display (preserved minimum if available, otherwise absolute minimum)
        if energy_stats["has_preserved_conformers"]:
            display_idx = energy_stats["preserved_min_idx"]
            display_mol = opt_mols[display_idx]
            display_type = "Lowest Energy (Topology & Stereochemistry Preserved)"
        else:
            display_idx = energy_stats["min_idx"]
            display_mol = opt_mols[display_idx]
            display_type = "Lowest Energy (Overall)"
        
        # Display results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(display_type)
            
            # 3D visualization
            mol_block = Chem.MolToMolBlock(display_mol)
            viewer = py3Dmol.view(width=600, height=400)
            viewer.addModel(mol_block, "mol")
            viewer.setStyle({'stick': {}})
            viewer.setBackgroundColor('white')
            viewer.zoomTo()
            showmol(viewer, height=400, width=600)
        
        with col2:
            st.subheader("Energy Analysis")
            
            # Energy statistics (relative to minimum)
            st.metric("Lowest Energy", "0.00 kcal/mol (reference)")
            st.metric("Highest Relative Energy", f"{energy_stats['max_relative_energy']:.2f} kcal/mol")
            st.metric("Average Relative Energy", f"{energy_stats['mean_relative_energy']:.2f} kcal/mol")
            st.metric("Energy Range", f"{energy_stats['energy_range']:.2f} kcal/mol")
            
            # Optimization metrics
            if isinstance(results, dict):
                if "opt_avg_energy_drop" in results:
                    st.metric("Avg Energy Drop", f"{results['opt_avg_energy_drop']:.2f} kcal/mol")
                if "opt_converged" in results:
                    st.metric("Optimization Success", f"{results['opt_converged']*100:.1f}%")
            
            # Topology preservation
            st.metric("Topology Preserved", f"{topology_results['topology_preserved_percentage']:.1f}%")
            
            # Stereochemistry preservation
            if stereo_results['has_stereochemistry']:
                st.metric("Stereochemistry Preserved", f"{stereo_results['stereo_preserved_percentage']:.1f}%")
            else:
                st.metric("Stereochemistry", "No R/S or E/Z centers")
        
        # Conformer table
        st.subheader("All Conformers")
        df_data = []
        for i, energy in enumerate(energies_kcal):
            # Check preservation status
            topology_ok = i < len(topology_results.get('topology_results', [])) and topology_results['topology_results'][i]
            stereo_ok = True  # Default for molecules without stereochemistry
            if stereo_results.get('has_stereochemistry', False):
                stereo_preserved_list = stereo_results.get('stereo_results', {}).get('preserved_stereo', [])
                stereo_ok = i < len(stereo_preserved_list) and stereo_preserved_list[i]
            
            df_data.append({
                'Conformer': i + 1,
                'Relative Energy (kcal/mol)': f"{energy - energy_stats['min_energy']:.2f}",
                'Topology OK': "✓" if topology_ok else "✗",
                'Stereochemistry OK': "✓" if stereo_ok else "✗" if stereo_results.get('has_stereochemistry', False) else "N/A",
                'Is Displayed': i == display_idx,
                'Is Global Min': i == energy_stats["min_idx"]
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)
        
        # Download SDF file
        st.subheader("Download Results")
        
        # Create SDF content using utility function
        sdf_content = create_sdf_content(opt_mols, energies_kcal, energy_stats['min_energy'])
        
        # Download button for all conformers
        st.download_button(
            label="📥 Download All Conformers (SDF)",
            data=sdf_content,
            file_name=safe_filename_from_smiles(smiles, "_conformers.sdf"),
            mime="chemical/x-mdl-sdfile",
            help="Download all generated conformers with energy information"
        )
        
        # Create SDF content for displayed conformer
        displayed_energy = energies_kcal[display_idx]
        displayed_sdf_content = create_sdf_content([display_mol], [displayed_energy], energy_stats['min_energy'])
        
        # Download button for displayed conformer
        download_label = "📥 Download Best Preserved Conformer (SDF)" if energy_stats["has_preserved_conformers"] else "📥 Download Lowest Energy Conformer (SDF)"
        download_suffix = "_best_preserved.sdf" if energy_stats["has_preserved_conformers"] else "_lowest_energy.sdf"
        
        st.download_button(
            label=download_label,
            data=displayed_sdf_content,
            file_name=safe_filename_from_smiles(smiles, download_suffix),
            mime="chemical/x-mdl-sdfile",
            help="Download the displayed conformer (best preserved or lowest energy)"
        )


# Footer
st.markdown("---")
st.markdown("**LoQI**: Low-energy QM Informed conformer generation with stereochemistry awareness")