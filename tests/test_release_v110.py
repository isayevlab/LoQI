"""Release routing without downloading or allocating a real model."""

import sys

import pytest

from loqi.cli import build_parser
from loqi.registry import MODELS


def test_release_registered_for_download_and_sampling():
    name = "loqi_flow_v1.1.0"
    assert MODELS[name].config == "loqi_flow.yaml"
    assert "1UOxDPX6u0n6Ij6mvqaT6PVmEREeBjJcS" in MODELS[name].url
    assert build_parser().parse_args(["download", "--model", name]).model == name
    assert build_parser().parse_args(["sample", "--smiles", "O", "--model", name, "--output", "out.sdf"]).model == name


@pytest.mark.parametrize("local", [False, True])
def test_sampling_script_defaults_to_release(monkeypatch, tmp_path, local):
    from scripts import sample_conformers as cli

    path = tmp_path / "loqi_flow_v1.1.0.ckpt"
    if local:
        path.write_bytes(b"test only")
    monkeypatch.setattr(cli, "RELEASE_CHECKPOINT", path)
    monkeypatch.setattr(sys, "argv", ["sample_conformers.py", "--input", "Cl", "--output", "unused.sdf"])

    def capture(name, **kwargs):
        assert name == (str(path) if local else "loqi_flow_v1.1.0")
        assert kwargs["config"] is None
        raise RuntimeError("release selected")

    monkeypatch.setattr(cli, "load_model", capture)
    with pytest.raises(RuntimeError, match="release selected"):
        cli.main()
