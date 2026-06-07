import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
import torch
from omegaconf import OmegaConf
import omegaconf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from megalodon.data.batch_preprocessor import BatchPreProcessor
from megalodon.metrics.molecule_metrics_aimnet2 import MoleculeAIMNet2Metrics
from megalodon.models.module import Graph3DInterpolantModel

from utils import (
    check_stereochemistry_preservation,
    check_topology_preservation,
    create_sdf_content,
    generate_conformers_batch,
    get_energy_statistics,
    render_molecule_svg,
    safe_filename_from_smiles,
    select_unique_with_irmsd,
    set_cfg_timesteps,
)

Chem.SetUseLegacyStereoPerception(True)

POSTPROCESS_NONE = "none"
POSTPROCESS_OPT = "optimization"
POSTPROCESS_OPT_IRMSD = "optimization + irmsd unique set selection"

EV_TO_KCAL_PER_MOL = 23.060547830619026
MODEL_OPTIONS = [
    "Diffusion",
    "Flow Matching",
]


@dataclass
class SidebarConfig:
    model_type: str
    postprocess_mode: str
    n_steps: int
    opt_fmax: float
    opt_max_nstep: int
    irmsd_rthr: float


@dataclass
class OptimizationResult:
    success: bool
    molecules: Optional[List[Chem.Mol]] = None
    energies_kcal: Optional[np.ndarray] = None
    metrics: Optional[Dict[str, Any]] = None
    warning: Optional[str] = None
    raw_error: Optional[str] = None


def is_optimization_enabled(postprocess_mode: str) -> bool:
    return postprocess_mode in {POSTPROCESS_OPT, POSTPROCESS_OPT_IRMSD}


def is_irmsd_enabled(postprocess_mode: str) -> bool:
    return postprocess_mode == POSTPROCESS_OPT_IRMSD


def clone_cfg(cfg):
    """Clone config to avoid mutating cached shared objects across Streamlit sessions."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


@st.cache_resource
def load_model(selected_model_type):
    torch.serialization.add_safe_globals([DictConfig, ListConfig])
    """Load model and config for selected model type."""
    # Hidden for now to keep the app surface small:
    # - Flow Liquid
    # - SEMLA Fast
    if selected_model_type == "Flow Matching":
        config_path = ROOT / "scripts/conf/loqi/loqi_flow.yaml"
        ckpt_path = ROOT / "data/loqi_flow.ckpt"
    else:
        config_path = ROOT / "scripts/conf/loqi/loqi.yaml"
        ckpt_path = ROOT / "data/loqi.ckpt"

    cfg = OmegaConf.load(config_path)
    cfg.data.dataset_root = str(ROOT / "data/chembl3d_stereo")
    if cfg.evaluation.energy_metrics_args is None:
        cfg.evaluation.energy_metrics_args = OmegaConf.create({})
    cfg.evaluation.energy_metrics_args.model_path = str(
        ROOT / "src/megalodon/metrics/aimnet2/cpcm_model/wb97m_cpcms_v2_0.jpt"
    )
    batch_preprocessor = BatchPreProcessor(cfg.data.aug_rotations, cfg.data.scale_coords)
    torch.serialization.add_safe_globals([
        omegaconf.dictconfig.DictConfig,
        omegaconf.listconfig.ListConfig,
    ])

    model = Graph3DInterpolantModel.load_from_checkpoint(
        str(ckpt_path),
        loss_params=cfg.loss,
        interpolant_params=cfg.interpolant,
        sampling_params=cfg.sample,
        batch_preprocessor=batch_preprocessor,
        weights_only=False,
    )
    model.batch_preprocessor = batch_preprocessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return model.to(device).eval(), cfg


def evaluate_energies(molecules, cfg, fmax, max_nstep) -> OptimizationResult:
    """Evaluate energies with AIMNet2 optimization."""
    aimnet_path = cfg.evaluation.energy_metrics_args.model_path
    if not os.path.exists(str(aimnet_path)):
        msg = "AIMNet2 model not found"
        return OptimizationResult(success=False, warning=msg, raw_error=msg)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    aimnet_batchsize = int(getattr(cfg.evaluation.energy_metrics_args, "batchsize", 100))
    energy_metrics = MoleculeAIMNet2Metrics(
        model_path=str(aimnet_path),
        batchsize=aimnet_batchsize,
        opt_metrics=True,
        opt_params={"fmax": float(fmax), "max_nstep": int(max_nstep)},
        device=device,
    )

    try:
        metrics, _, opt_mols, opt_energies = energy_metrics(
            molecules, reference_molecules=None, return_molecules=True
        )
        energies_kcal = opt_energies.cpu().numpy() * EV_TO_KCAL_PER_MOL
        return OptimizationResult(
            success=True,
            molecules=opt_mols,
            energies_kcal=energies_kcal,
            metrics=metrics,
        )
    except Exception as exc:
        raw = str(exc)
        if isinstance(exc, UnboundLocalError) and "converged" in raw:
            msg = (
                "Optimization failed in the AIMNet2 optimizer backend (internal convergence-state bug). "
                "This is usually a downstream symptom after optimization did not finish cleanly for one or more "
                "conformers."
            )
        else:
            msg = "Optimization failed. Continuing without optimization."
        return OptimizationResult(success=False, warning=msg, raw_error=raw)


def render_2d_preview(smiles: str) -> None:
    st.header("Molecule Visualization")
    if not smiles:
        return

    try:
        svg = render_molecule_svg(smiles)
        if svg:
            svg_clean = svg.replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")
            st.markdown(
                "<div style='background:white;border:1px solid #e5e7eb;border-radius:12px;"
                "padding:0.5rem 0.75rem;'>"
                + svg_clean.replace("width='900px'", "width='100%'").replace("height='320px'", "")
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.error("Invalid SMILES string")
    except Exception as exc:
        st.error(f"Error drawing 2D structure: {exc}")


def render_3d_molecule(molecule: Chem.Mol, title: str) -> None:
    st.subheader(title)
    mol_block = Chem.MolToMolBlock(molecule)
    viewer = py3Dmol.view(width=600, height=400)
    viewer.addModel(mol_block, "mol")
    viewer.setStyle({"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}})
    viewer.setBackgroundColor("white")
    viewer.zoomTo()
    components.html(viewer._make_html(), height=400, width=600, scrolling=False)


def build_sidebar_config() -> SidebarConfig:
    st.sidebar.header("Configuration")
    if "selected_model_type" not in st.session_state:
        st.session_state.selected_model_type = MODEL_OPTIONS[0]
    if "postprocess_mode" not in st.session_state:
        st.session_state.postprocess_mode = POSTPROCESS_OPT

    def _sync_postprocess_default() -> None:
        if st.session_state.postprocess_mode == POSTPROCESS_NONE:
            st.session_state.postprocess_mode = POSTPROCESS_OPT

    model_type = st.sidebar.selectbox(
        "Model",
        MODEL_OPTIONS,
        key="selected_model_type",
        on_change=_sync_postprocess_default,
    )
    postprocess_mode = st.sidebar.selectbox(
        "Postprocessing",
        [POSTPROCESS_NONE, POSTPROCESS_OPT, POSTPROCESS_OPT_IRMSD],
        key="postprocess_mode",
    )

    if model_type == "Flow Matching":
        n_steps = st.sidebar.slider("Sampling Steps", min_value=1, max_value=100, value=25)
    else:
        n_steps = 25
        st.sidebar.text_input("Sampling Steps (Diffusion)", value=str(n_steps), disabled=True)

    if is_optimization_enabled(postprocess_mode):
        st.sidebar.subheader("Optimization Parameters")
        opt_fmax = st.sidebar.number_input("fmax", min_value=1e-5, max_value=1e-1, value=2e-3, format="%.5f")
        opt_max_nstep = st.sidebar.number_input("max_nstep", min_value=100, max_value=20000, value=3000, step=100)
    else:
        opt_fmax = 2e-3
        opt_max_nstep = 3000

    if is_irmsd_enabled(postprocess_mode):
        st.sidebar.subheader("iRMSD Parameters")
        irmsd_rthr = st.sidebar.number_input("rthr (A)", min_value=0.01, max_value=2.0, value=0.125, format="%.3f")
    else:
        irmsd_rthr = 0.125

    if st.sidebar.button("Clear Model Cache"):
        st.cache_resource.clear()
        st.sidebar.success("Cache cleared. The model will reload on next generation.")

    return SidebarConfig(
        model_type=model_type,
        postprocess_mode=postprocess_mode,
        n_steps=int(n_steps),
        opt_fmax=float(opt_fmax),
        opt_max_nstep=int(opt_max_nstep),
        irmsd_rthr=float(irmsd_rthr),
    )


def build_conformer_rows(working_mols, energy_stats, energies_kcal, topology_results_display, stereo_results_display, display_idx):
    rows = []
    for i in range(len(working_mols)):
        rel_energy = "N/A"
        if energy_stats is not None:
            rel_energy = f"{energies_kcal[i] - energy_stats['min_energy']:.2f}"

        topology_ok = (
            i < len(topology_results_display.get("topology_results", []))
            and topology_results_display["topology_results"][i]
        )
        stereo_ok = True
        if stereo_results_display.get("has_stereochemistry", False):
            preserved = stereo_results_display.get("stereo_results", {}).get("preserved_stereo", [])
            stereo_ok = i < len(preserved) and preserved[i]

        rows.append(
            {
                "Conformer": i + 1,
                "Relative Energy (kcal/mol)": rel_energy,
                "Topology OK": "✓" if topology_ok else "✗",
                "Stereochemistry OK": "✓" if stereo_ok else "✗"
                if stereo_results_display.get("has_stereochemistry", False)
                else "N/A",
                "Is Displayed": i == display_idx,
                "Is Best of Generated": energy_stats is not None and i == energy_stats["min_idx"],
            }
        )

    return rows


def render_generation_results(result_data, smiles: str) -> None:
    generated_mols = result_data["generated_mols"]
    working_mols = result_data["working_mols"]
    working_refs = result_data["working_refs"]
    metrics_mols = result_data["metrics_mols"]
    metrics_refs = result_data["metrics_refs"]
    energies_kcal = result_data["energies_kcal"]
    optimization_metrics = result_data["optimization_metrics"]
    topology_results_metrics = result_data["topology_results_metrics"]
    stereo_results_metrics = result_data["stereo_results_metrics"]
    topology_results_display = result_data["topology_results_display"]
    stereo_results_display = result_data["stereo_results_display"]
    energy_stats = result_data["energy_stats"]
    gen_time_per_structure_s = result_data["gen_time_per_structure_s"]

    st.success(f"Generated {len(generated_mols)} conformers successfully.")
    st.metric("Generation Time / Structure", f"{gen_time_per_structure_s:.4f} s")

    if energy_stats is not None:
        if energy_stats["has_preserved_conformers"]:
            recommended_display_idx = energy_stats["preserved_min_idx"]
        else:
            recommended_display_idx = energy_stats["min_idx"]
    else:
        recommended_display_idx = 0

    slider_key = "displayed_conformer"
    if (
        slider_key not in st.session_state
        or st.session_state[slider_key] < 1
        or st.session_state[slider_key] > len(working_mols)
    ):
        st.session_state[slider_key] = recommended_display_idx + 1

    selected_conformer = st.slider(
        "Displayed Conformer",
        min_value=1,
        max_value=len(working_mols),
        key=slider_key,
        help="Scroll through all generated conformers. Starts on the recommended best conformer.",
    )
    display_idx = selected_conformer - 1
    if energy_stats is not None:
        if display_idx == energy_stats["min_idx"]:
            display_type = "Best of Generated (Overall)"
        elif energy_stats["has_preserved_conformers"] and display_idx == energy_stats["preserved_min_idx"]:
            display_type = "Lowest Energy Preserved"
        else:
            display_type = f"Conformer {selected_conformer}"
    else:
        display_type = f"Conformer {selected_conformer}"

    display_mol = working_mols[display_idx]

    vis_col, stats_col = st.columns([2, 1])
    with vis_col:
        render_3d_molecule(display_mol, display_type)
        if energy_stats is not None:
            st.caption(f"Recommended starting conformer: {recommended_display_idx + 1}.")

    with stats_col:
        st.subheader("Analysis")
        if energy_stats is not None:
            st.metric("Highest Relative Energy", f"{energy_stats['max_relative_energy']:.2f} kcal/mol")
            st.metric("Average Relative Energy", f"{energy_stats['mean_relative_energy']:.2f} kcal/mol")
        else:
            st.metric("Energies", "Not computed")

        if isinstance(optimization_metrics, dict):
            if "opt_total_time" in optimization_metrics:
                avg_opt_time = float(optimization_metrics["opt_total_time"]) / max(len(generated_mols), 1)
                st.metric("Avg Optimization Time / Structure", f"{avg_opt_time:.4f} s")
            if "opt_avg_energy_drop" in optimization_metrics:
                st.metric("Avg Energy Drop", f"{optimization_metrics['opt_avg_energy_drop']:.2f} kcal/mol")
            if "opt_converged" in optimization_metrics:
                st.metric("Optimization Success", f"{optimization_metrics['opt_converged'] * 100:.1f}%")

        st.metric("Topology Preserved", f"{topology_results_metrics['topology_preserved_percentage']:.1f}%")
        if stereo_results_metrics["has_stereochemistry"]:
            st.metric("Stereochemistry Preserved", f"{stereo_results_metrics['stereo_preserved_percentage']:.1f}%")
        else:
            st.metric("Stereochemistry", "No R/S or E/Z centers")

    st.subheader("All Conformers")
    rows = build_conformer_rows(
        working_mols,
        energy_stats,
        energies_kcal,
        topology_results_display,
        stereo_results_display,
        display_idx,
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Download Results")
    sdf_content = create_sdf_content(
        working_mols,
        energies_kcal,
        energy_stats["min_energy"] if energy_stats is not None else None,
    )
    st.download_button(
        label="📥 Download All Conformers (SDF)",
        data=sdf_content,
        file_name=safe_filename_from_smiles(smiles, "_conformers.sdf"),
        mime="chemical/x-mdl-sdfile",
    )

    single_energy = [energies_kcal[display_idx]] if energies_kcal is not None else None
    displayed_sdf_content = create_sdf_content(
        [display_mol],
        single_energy,
        energy_stats["min_energy"] if energy_stats is not None else None,
    )
    is_preserved = energy_stats is not None and energy_stats["has_preserved_conformers"]
    suffix = "_best_preserved.sdf" if is_preserved else "_displayed.sdf"
    label = "📥 Download Best Preserved Conformer (SDF)" if is_preserved else "📥 Download Displayed Conformer (SDF)"
    st.download_button(
        label=label,
        data=displayed_sdf_content,
        file_name=safe_filename_from_smiles(smiles, suffix),
        mime="chemical/x-mdl-sdfile",
    )


st.set_page_config(page_title="LoQI Conformer Generator", page_icon="🧬", layout="wide")
st.title("🧬 LoQI: Low-Energy QM Informed Conformer Generator")
st.markdown("Generate and visualize low-energy molecular conformers with quantum mechanical accuracy")

sidebar_cfg = build_sidebar_config()

left, right = st.columns([1, 2])
with left:
    st.header("Input")
    smiles = st.text_input(
        "Enter SMILES",
        "[H]c1c([H])c([H])c2c(c1[H])C(=O)N([C@@]1([H])C(=O)N([H])C(=O)C([H])([H])C1([H])[H])C2=O",
        help="Thalidomide example with stereochemistry",
    )
    n_confs = st.slider("Number of conformers", min_value=1, max_value=20, value=10)
    generate_button = st.button("Generate Conformers", type="primary")

with right:
    render_2d_preview(smiles)


if generate_button and smiles:
    mol_for_warning = Chem.MolFromSmiles(smiles)
    if mol_for_warning is not None and len(Chem.GetMolFrags(mol_for_warning)) > 1:
        st.warning(
            "Disconnected fragments detected. Generation will run in permissive mode "
            "(experimental for dimers/multimers)."
        )

    with st.spinner(f"Loading {sidebar_cfg.model_type} model..."):
        model, base_cfg = load_model(sidebar_cfg.model_type)
        run_cfg = clone_cfg(base_cfg)
        run_cfg = set_cfg_timesteps(run_cfg, sidebar_cfg.n_steps)

    with st.spinner(f"Generating {n_confs} conformers ({sidebar_cfg.n_steps} steps)..."):
        generation_batchsize = int(
            getattr(run_cfg.data, "inference_batch_size", getattr(run_cfg.data, "batch_size", n_confs))
        )
        generated_mols, reference_mols, gen_time_per_structure_s, error = generate_conformers_batch(
            smiles, model, run_cfg, n_confs, generation_batch_size=generation_batchsize
        )

    if error:
        if "Disconnected fragments are not supported" in str(error):
            st.warning(
                "Disconnected fragments detected. Generation is experimental for dimers/multimers."
            )
        else:
            st.error(f"Error generating conformers: {error}")
    elif generated_mols:
        working_mols = generated_mols
        working_refs = reference_mols
        metrics_mols = generated_mols
        metrics_refs = reference_mols
        energies_kcal = None
        optimization_metrics = None

        if is_optimization_enabled(sidebar_cfg.postprocess_mode):
            with st.spinner("Running optimization..."):
                opt_result = evaluate_energies(
                    working_mols, run_cfg, sidebar_cfg.opt_fmax, sidebar_cfg.opt_max_nstep
                )

            if not opt_result.success:
                st.warning(opt_result.warning)
                st.info("Continuing with generated conformers without optimization.")
                if opt_result.raw_error:
                    with st.expander("Optimization debug details"):
                        st.code(opt_result.raw_error)
            else:
                working_mols = opt_result.molecules
                metrics_mols = opt_result.molecules
                energies_kcal = opt_result.energies_kcal
                optimization_metrics = opt_result.metrics

        if is_irmsd_enabled(sidebar_cfg.postprocess_mode):
            with st.spinner("Running iRMSD unique-set selection..."):
                unique_mols, selected_indices, irmsd_error = select_unique_with_irmsd(
                    working_mols, rthr=sidebar_cfg.irmsd_rthr
                )
            if irmsd_error:
                st.error(irmsd_error)
                st.stop()
            working_mols = unique_mols
            working_refs = [working_refs[i] for i in selected_indices]
            if energies_kcal is not None:
                energies_kcal = np.array([energies_kcal[i] for i in selected_indices])
            st.info(f"Unique conformers after iRMSD pruning: {len(working_mols)}")

        with st.spinner("Checking topology and stereochemistry preservation..."):
            topology_results_metrics = check_topology_preservation(metrics_mols)
            stereo_results_metrics = check_stereochemistry_preservation(metrics_mols, metrics_refs)
            topology_results_display = check_topology_preservation(working_mols)
            stereo_results_display = check_stereochemistry_preservation(working_mols, working_refs)

        if energies_kcal is not None:
            energy_stats = get_energy_statistics(energies_kcal, topology_results_display, stereo_results_display)
            if energy_stats["has_preserved_conformers"]:
                display_idx = energy_stats["preserved_min_idx"]
                display_type = "Lowest Energy Preserved"
            else:
                display_idx = energy_stats["min_idx"]
                display_type = "Best of Generated (Overall)"
        else:
            energy_stats = None
            energy_stats = None

        st.session_state.generated_results = {
            "smiles": smiles,
            "generated_mols": generated_mols,
            "working_mols": working_mols,
            "working_refs": working_refs,
            "metrics_mols": metrics_mols,
            "metrics_refs": metrics_refs,
            "energies_kcal": energies_kcal,
            "optimization_metrics": optimization_metrics,
            "topology_results_metrics": topology_results_metrics,
            "stereo_results_metrics": stereo_results_metrics,
            "topology_results_display": topology_results_display,
            "stereo_results_display": stereo_results_display,
            "energy_stats": energy_stats,
            "gen_time_per_structure_s": gen_time_per_structure_s,
        }
        if energy_stats is not None:
            if energy_stats["has_preserved_conformers"]:
                st.session_state.displayed_conformer = energy_stats["preserved_min_idx"] + 1
            else:
                st.session_state.displayed_conformer = energy_stats["min_idx"] + 1
        else:
            st.session_state.displayed_conformer = 1

if "generated_results" in st.session_state:
    render_generation_results(st.session_state.generated_results, st.session_state.generated_results["smiles"])

st.markdown("---")
st.markdown("**LoQI**: Low-energy QM Informed conformer generation with stereochemistry awareness")
