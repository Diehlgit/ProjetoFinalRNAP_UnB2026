"""Download de fotos com nova tentativa, escrita atômica e SHA-256."""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(session: requests.Session, url: str, dest: Path, retries: int = 4) -> str | None:
    """Baixa `url` para `dest` (se ainda não existe). Retorna mensagem de erro ou None."""
    if dest.exists():
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    err = ""
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=60)
            if r.status_code == 404 or r.status_code == 403:
                return f"HTTP {r.status_code}"
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(r.content)
            tmp.replace(dest)
            return None
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
            err = str(e)
            time.sleep(2 ** (attempt + 1))
    return err or "falha desconhecida"


def download_many(
    session: requests.Session, jobs: list[tuple[str, Path]], workers: int
) -> dict[Path, str]:
    """Baixa em paralelo; retorna {destino: erro} das falhas."""
    from tqdm import tqdm

    failures: dict[Path, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download, session, url, dest): dest for url, dest in jobs}
        for fut in tqdm(futures, desc="fotos", unit="foto"):
            err = fut.result()
            if err:
                failures[futures[fut]] = err
    return failures
