"""Checkpoint registry and verified download cache."""

import hashlib
import re
from pathlib import Path

import pytest

from loqi import registry
from loqi.api import bundled_config_path
from loqi.registry import MODELS, ModelEntry, checkpoint_path, default_cache_dir

PAYLOAD = b"not a real checkpoint\n" * 4096


def test_models_table_is_well_formed():
    assert set(MODELS) == {"loqi", "loqi_flow"}
    for entry in MODELS.values():
        assert re.fullmatch(r"[0-9a-f]{64}", entry.sha256)
        assert entry.url.startswith("https://ndownloader.figshare.com/files/")
        assert bundled_config_path(entry.config).is_file()
    assert len({e.sha256 for e in MODELS.values()}) == len(MODELS)
    assert len({e.url for e in MODELS.values()}) == len(MODELS)


def test_default_cache_dir_honours_environment(monkeypatch, tmp_path):
    monkeypatch.setenv(registry.ENV_CACHE_DIR, str(tmp_path))
    assert default_cache_dir() == tmp_path
    monkeypatch.delenv(registry.ENV_CACHE_DIR)
    assert default_cache_dir() == Path.home() / ".cache" / "loqi"


def test_checkpoint_path_accepts_local_file(tmp_path):
    local = tmp_path / "model.ckpt"
    local.write_bytes(b"x")
    assert checkpoint_path(local) == local
    assert checkpoint_path(str(local)) == local


def test_checkpoint_path_rejects_unknown_name(tmp_path):
    with pytest.raises(FileNotFoundError, match="loqi"):
        checkpoint_path("not-a-model", cache_dir=tmp_path)


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    source.write_bytes(PAYLOAD)
    entry = ModelEntry(url=source.as_uri(), sha256=hashlib.sha256(PAYLOAD).hexdigest(), config="loqi.yaml")
    monkeypatch.setitem(MODELS, "fake", entry)
    return entry, tmp_path / "cache"


def test_download_verifies_renames_and_caches(fake_model, monkeypatch):
    _, cache = fake_model
    path = checkpoint_path("fake", cache_dir=cache, progress=False)
    assert path == cache / "fake.ckpt"
    assert path.read_bytes() == PAYLOAD
    assert list(cache.glob("*.part")) == []

    def no_download(*args, **kwargs):
        raise AssertionError("a verified cached file must not be downloaded again")

    monkeypatch.setattr(registry, "download_file", no_download)
    assert checkpoint_path("fake", cache_dir=cache, progress=False) == path


def test_download_uses_env_cache_dir(fake_model, monkeypatch, tmp_path):
    monkeypatch.setenv(registry.ENV_CACHE_DIR, str(tmp_path / "env-cache"))
    path = checkpoint_path("fake", progress=False)
    assert path == tmp_path / "env-cache" / "fake.ckpt"


def test_sha256_mismatch_discards_download(fake_model, monkeypatch):
    entry, cache = fake_model
    monkeypatch.setitem(MODELS, "fake", ModelEntry(entry.url, "0" * 64, entry.config))
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        checkpoint_path("fake", cache_dir=cache, progress=False)
    assert not (cache / "fake.ckpt").exists()
    assert list(cache.glob("*.part")) == []


def test_corrupted_cached_file_is_replaced(fake_model):
    _, cache = fake_model
    cache.mkdir()
    (cache / "fake.ckpt").write_bytes(b"corrupted")
    with pytest.warns(UserWarning, match="SHA-256"):
        path = checkpoint_path("fake", cache_dir=cache, progress=False)
    assert path.read_bytes() == PAYLOAD
