"""Registro de metadados de execução para reprodutibilidade.

Cada experimento deve salvar, junto com seus resultados, o hash do commit do
código, as versões dos pacotes e informações do hardware utilizado.
"""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Fixa as sementes de Python, NumPy e PyTorch (quando disponíveis)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _git(*args: str, cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect(repo_dir: str | Path | None = None) -> dict[str, Any]:
    """Coleta commit, estado do repositório, versões de pacotes e hardware."""
    repo = Path(repo_dir) if repo_dir else Path(__file__).resolve().parents[3]
    status = _git("status", "--porcelain", cwd=repo)
    meta: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD", cwd=repo),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        meta["torch"] = torch.__version__
        meta["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            meta["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        meta["torch"] = None
    try:
        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        )
        meta["pip_freeze"] = freeze.stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        meta["pip_freeze"] = None
    return meta


def save(path: str | Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Salva os metadados (mais campos extras, como a configuração) em JSON."""
    meta = collect()
    if extra:
        meta.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta
