"""Argument parsing and early exits of the ``loqi`` command line interface."""

from pathlib import Path

import pytest

from loqi.cli import _read_smiles_file, build_parser, main


def test_sample_parser_defaults_and_repeatable_smiles():
    args = build_parser().parse_args(
        ["sample", "--smiles", "CCO", "--smiles", "c1ccccc1", "--n-confs", "3", "--output", "out.sdf"]
    )
    assert args.command == "sample"
    assert args.smiles == ["CCO", "c1ccccc1"]
    assert args.n_confs == 3
    assert args.output == Path("out.sdf")
    assert args.model == "loqi"
    assert args.device is None
    assert args.seed == 42
    assert args.steps is None
    assert args.batch_atoms is None
    assert args.add_hs is True
    assert args.input is None


def test_sample_parser_options():
    args = build_parser().parse_args(
        ["sample", "--input", "mols.smi", "--output", "o.sdf", "--model", "loqi_flow", "--device", "cpu",
         "--seed", "1", "--steps", "50", "--batch-atoms", "1000", "--no-add-hs"]
    )  # fmt: skip
    assert args.input == Path("mols.smi")
    assert args.model == "loqi_flow"
    assert args.device == "cpu"
    assert (args.seed, args.steps, args.batch_atoms, args.add_hs) == (1, 50, 1000, False)


def test_sample_requires_output():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["sample", "--smiles", "CCO"])


def test_subcommand_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_download_parser():
    args = build_parser().parse_args(["download", "--model", "loqi_flow", "--cache-dir", "/tmp/c"])
    assert args.command == "download"
    assert args.model == "loqi_flow"
    assert args.cache_dir == Path("/tmp/c")
    assert build_parser().parse_args(["download"]).model == "loqi"


def test_download_unknown_model_exits_2(capsys):
    assert main(["download", "--model", "nope"]) == 2
    assert "unknown model" in capsys.readouterr().err


def test_sample_without_smiles_exits_2(tmp_path, capsys):
    assert main(["sample", "--output", str(tmp_path / "out.sdf")]) == 2
    assert "--smiles" in capsys.readouterr().err


def test_sample_with_only_invalid_smiles_exits_1(tmp_path, capsys):
    assert main(["sample", "--smiles", "C(C", "--output", str(tmp_path / "out.sdf")]) == 1
    err = capsys.readouterr().err
    assert "skipping" in err and "no valid SMILES" in err
    assert not (tmp_path / "out.sdf").exists()


def test_read_smiles_file_skips_comments_and_names(tmp_path):
    path = tmp_path / "in.smi"
    path.write_text("# header\nCCO ethanol\n\nc1ccccc1\tbenzene\n")
    assert _read_smiles_file(path) == ["CCO", "c1ccccc1"]
