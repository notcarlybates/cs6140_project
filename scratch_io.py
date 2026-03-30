"""
Transparent I/O helper for paths on the remote /scratch filesystem.

Any path containing "scratch" is read/written via rsync over SSH.
All other paths use the local filesystem directly.

Requires SSH key-based auth to be configured for REMOTE.
"""

import os
import subprocess
import tempfile
import numpy as np
import polars as pl
import joblib

REMOTE = "bates.car@xfer.discovery.neu.edu"


def _is_scratch(path: str) -> bool:
    return "scratch" in path


def _ssh(cmd: str) -> str:
    result = subprocess.run(["ssh", REMOTE, cmd], capture_output=True, text=True, check=True)
    return result.stdout


def _rsync_download(remote_path: str, local_path: str) -> None:
    subprocess.run(["rsync", "-az", f"{REMOTE}:{remote_path}", local_path], check=True)


def _rsync_upload(local_path: str, remote_path: str) -> None:
    subprocess.run(["rsync", "-az", local_path, f"{REMOTE}:{remote_path}"], check=True)


def exists(path: str) -> bool:
    if _is_scratch(path):
        result = subprocess.run(
            ["ssh", REMOTE, f"test -e {path}"],
            capture_output=True,
        )
        return result.returncode == 0
    else:
        return os.path.exists(path)


def download_to_temp(path: str) -> str:
    """Download a scratch file to a local temp file. Returns local path. Caller must unlink if path differs."""
    if _is_scratch(path):
        suffix = os.path.splitext(path)[1] or ".tmp"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            tmp = f.name
        _rsync_download(path, tmp)
        return tmp
    return path


def makedirs(path: str) -> None:
    if _is_scratch(path):
        _ssh(f"mkdir -p {path}")
    else:
        os.makedirs(path, exist_ok=True)


def list_dir(path: str) -> list[str]:
    if _is_scratch(path):
        output = _ssh(f"ls {path}")
        return [f for f in output.strip().split("\n") if f]
    else:
        return os.listdir(path)


def read_csv(path: str) -> pl.DataFrame:
    if _is_scratch(path):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            tmp = f.name
        try:
            _rsync_download(path, tmp)
            return pl.read_csv(tmp)
        finally:
            os.unlink(tmp)
    else:
        return pl.read_csv(path)


def write_csv(df: pl.DataFrame, path: str) -> None:
    if _is_scratch(path):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            tmp = f.name
        try:
            df.write_csv(tmp)
            _rsync_upload(tmp, path)
        finally:
            os.unlink(tmp)
    else:
        df.write_csv(path)


def load_npy(path: str, allow_pickle: bool = False) -> np.ndarray:
    if _is_scratch(path):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            tmp = f.name
        try:
            _rsync_download(path, tmp)
            return np.load(tmp, allow_pickle=allow_pickle)
        finally:
            os.unlink(tmp)
    else:
        return np.load(path, allow_pickle=allow_pickle)


def save_npy(data, path: str, allow_pickle: bool = False) -> None:
    if _is_scratch(path):
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            tmp = f.name
        try:
            np.save(tmp, data, allow_pickle=allow_pickle)
            _rsync_upload(tmp, path)
        finally:
            os.unlink(tmp)
    else:
        np.save(path, data, allow_pickle=allow_pickle)


def joblib_dump(obj, path: str) -> None:
    if _is_scratch(path):
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            tmp = f.name
        try:
            joblib.dump(obj, tmp)
            _rsync_upload(tmp, path)
        finally:
            os.unlink(tmp)
    else:
        joblib.dump(obj, path)
