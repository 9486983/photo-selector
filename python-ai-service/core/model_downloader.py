from __future__ import annotations

import hashlib
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_MODEL_CACHE_DIR: Path | None = None


def get_model_cache_dir() -> Path:
    global _MODEL_CACHE_DIR
    if _MODEL_CACHE_DIR is not None:
        return _MODEL_CACHE_DIR

    candidates = [
        Path(__file__).resolve().parent.parent / "models",
        Path.home() / ".cache" / "photo-selector" / "models",
    ]
    for c in candidates:
        c.mkdir(parents=True, exist_ok=True)
        try:
            (c / ".write_test").touch()
            (c / ".write_test").unlink()
            _MODEL_CACHE_DIR = c
            return c
        except OSError:
            continue
    _MODEL_CACHE_DIR = candidates[0]
    _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _MODEL_CACHE_DIR


def set_model_cache_dir(path: str | Path) -> None:
    global _MODEL_CACHE_DIR
    _MODEL_CACHE_DIR = Path(path)
    _MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


async def ensure_model(name: str, url: str, expected_md5: str | None = None) -> Path:
    cache_dir = get_model_cache_dir()
    dest = cache_dir / name

    if dest.exists():
        if expected_md5:
            actual = hashlib.md5(dest.read_bytes()).hexdigest()
            if actual == expected_md5:
                return dest
            logger.warning("MD5 mismatch for %s, re-downloading", name)
        else:
            return dest

    logger.info("Downloading model %s from %s", name, url)
    try:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=600)) as resp:
                resp.raise_for_status()
                tmp = Path(tempfile.mktemp(dir=str(cache_dir)))
                tmp.write_bytes(await resp.read())
                tmp.rename(dest)
    except ImportError:
        import urllib.request

        tmp = Path(tempfile.mktemp(dir=str(cache_dir)))
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
    except Exception:
        if dest.exists():
            return dest
        raise

    if expected_md5:
        actual = hashlib.md5(dest.read_bytes()).hexdigest()
        if actual != expected_md5:
            dest.unlink(missing_ok=True)
            raise ValueError(f"MD5 mismatch for {name}: expected {expected_md5}, got {actual}")

    logger.info("Model %s downloaded to %s", name, dest)
    return dest


def get_model_path(name: str, search_paths: list[Path] | None = None) -> Path | None:
    paths = search_paths or [
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent / "models",
        get_model_cache_dir(),
    ]
    for p in paths:
        candidate = p / name
        if candidate.exists():
            return candidate
        candidate = p / f"{name}.pt"
        if candidate.exists():
            return candidate
        candidate = p / f"{name}.onnx"
        if candidate.exists():
            return candidate
    return None
