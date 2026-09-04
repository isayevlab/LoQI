"""Registry of released LoQI checkpoints and a small verified download cache."""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

# The checkpoints are the files published with the KiltHub record
# "LoQI: Scalable Low-Energy Molecular Conformer Generation with Quantum Mechanical Accuracy",
# https://doi.org/10.1184/R1/31441570 (MIT license). The URLs are the record's figshare
# file downloads for loqi.ckpt and loqi_flow.ckpt.
KILTHUB_DOI = "10.1184/R1/31441570"

ENV_CACHE_DIR = "LOQI_CACHE_DIR"
_CHUNK_SIZE = 1 << 20


@dataclass(frozen=True)
class ModelEntry:
    """A downloadable checkpoint together with the bundled inference config that matches it."""

    url: str
    sha256: str
    config: str


MODELS: dict[str, ModelEntry] = {
    "loqi": ModelEntry(
        url="https://ndownloader.figshare.com/files/62280784",
        sha256="5ebf59836216a4249f5d856c6f3c750d86f9651acfb8745640f13ffaaeb0c007",
        config="loqi.yaml",
    ),
    "loqi_flow": ModelEntry(
        url="https://ndownloader.figshare.com/files/62280790",
        sha256="a6b44c07e80d4d020bdc971fe97da17d127c22164deb4608873cdc6719ece1ba",
        config="loqi_flow.yaml",
    ),
}


def default_cache_dir() -> Path:
    """Checkpoint cache directory: ``$LOQI_CACHE_DIR`` if set, else ``~/.cache/loqi``."""
    env = os.environ.get(ENV_CACHE_DIR)
    return Path(env).expanduser() if env else Path.home() / ".cache" / "loqi"


def sha256sum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, dest: str | Path, *, sha256: str | None = None, progress: bool = True) -> Path:
    """Stream ``url`` to ``dest``, verifying the SHA-256 digest before the file is moved into place.

    The download goes to a temporary file in the destination directory and is renamed atomically,
    so an interrupted or corrupted download never leaves a partial file at ``dest``.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f"{dest.name}.", suffix=".part")
    tmp = Path(tmp_name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "loqi"})
        with os.fdopen(fd, "wb") as fh, urllib.request.urlopen(request) as response:
            length = response.headers.get("Content-Length")
            with tqdm(
                total=int(length) if length else None,
                unit="B",
                unit_scale=True,
                desc=dest.name,
                disable=not progress,
            ) as bar:
                while chunk := response.read(_CHUNK_SIZE):
                    fh.write(chunk)
                    digest.update(chunk)
                    bar.update(len(chunk))
        if sha256 is not None and digest.hexdigest() != sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {url}: expected {sha256}, got {digest.hexdigest()}. The download was discarded."
            )
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def checkpoint_path(
    name_or_path: str | Path = "loqi", cache_dir: str | Path | None = None, *, progress: bool = True
) -> Path:
    """Return a local path to a checkpoint, downloading a registered model into the cache if needed.

    ``name_or_path`` is either a key of :data:`MODELS` or a path to an existing checkpoint file.
    Registered models are stored as ``<cache_dir>/<name>.ckpt``; an existing file is re-used only if
    its SHA-256 digest matches the registry, otherwise it is downloaded again.
    """
    name = str(name_or_path)
    if name in MODELS:
        entry = MODELS[name]
        directory = Path(cache_dir).expanduser() if cache_dir is not None else default_cache_dir()
        dest = directory / f"{name}.ckpt"
        if dest.is_file():
            if sha256sum(dest) == entry.sha256:
                return dest
            warnings.warn(f"{dest} does not match the expected SHA-256 digest; downloading it again.", stacklevel=2)
        return download_file(entry.url, dest, sha256=entry.sha256, progress=progress)

    path = Path(name).expanduser()
    if path.is_file():
        return path
    raise FileNotFoundError(f"{name!r} is neither a registered model ({', '.join(MODELS)}) nor an existing file.")
