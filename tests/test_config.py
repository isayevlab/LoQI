"""Bundled inference configs."""

import pytest
from omegaconf import OmegaConf

from loqi.api import bundled_config_path, load_config
from loqi.registry import MODELS


@pytest.mark.parametrize("name", ["loqi.yaml", "loqi_flow.yaml"])
def test_bundled_config_loads_with_training_paths_neutralised(name):
    assert bundled_config_path(name).is_file()
    cfg = load_config(name)
    assert cfg.sample.node_distribution is None
    assert cfg.data.dataset_root is None
    assert cfg.evaluation.energy_metrics_args.model_path is None
    assert cfg.wandb_params.mode == "disabled"
    assert cfg.interpolant.timesteps == 25
    assert cfg.dynamics.model_name == "megav3conf"
    assert cfg.data.aug_rotations is False
    assert cfg.data.scale_coords == 1.0
    assert cfg.data.inference_batch_size == 150


def test_registry_entries_point_to_bundled_configs():
    for entry in MODELS.values():
        load_config(entry.config)


def test_diffusion_and_flow_configs_differ():
    loqi = load_config("loqi.yaml")
    flow = load_config("loqi_flow.yaml")
    assert loqi.interpolant.variables[0].interpolant_type == "continuous_diffusion"
    assert flow.interpolant.variables[0].interpolant_type == "continuous_flow_matching"
    assert loqi.interpolant.time_type == "discrete"
    assert flow.interpolant.time_type == "continuous"


def test_load_config_from_path_forces_null_node_distribution(tmp_path):
    path = tmp_path / "custom.yaml"
    OmegaConf.save(OmegaConf.create({"sample": {"node_distribution": "missing.pickle"}, "x": 1}), path)
    cfg = load_config(path)
    assert cfg.sample.node_distribution is None
    assert cfg.x == 1

    bare = tmp_path / "bare.yaml"
    OmegaConf.save(OmegaConf.create({"x": 1}), bare)
    assert load_config(bare).sample.node_distribution is None


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError, match="loqi.yaml"):
        load_config("does-not-exist.yaml")
