"""Coleta de observações do iNaturalist (API v1) e extração das fotos elegíveis.

A licença é verificada **foto a foto** (regra 3 do CLAUDE.md): o filtro `photo_license`
da API só garante que a observação tem *alguma* foto com licença aceita.
"""

from __future__ import annotations

import gzip
import json
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import requests

PER_PAGE = 200


def make_session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = user_agent
    return s


def get_json(
    session: requests.Session, url: str, params: dict | None = None, retries: int = 5
) -> dict:
    """GET com nova tentativa e espera exponencial em 429/5xx e erros de rede."""
    for attempt in range(retries):
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise AssertionError("inalcançável")


def query_params(cfg: dict[str, Any]) -> dict[str, Any]:
    """Parâmetros fixos da consulta, derivados de `configs/dataset.yaml` (seção inaturalist)."""
    p = {
        "place_id": cfg["place_id"],
        "taxon_id": cfg["taxon_id"],
        "quality_grade": cfg["quality_grade"],
        "photo_license": ",".join(cfg["photo_licenses"]),
        "created_d2": str(cfg["created_before"]),
        "order_by": "id",
        "order": "asc",
        "per_page": PER_PAGE,
    }
    return p


def iter_observations(cfg: dict[str, Any], session: requests.Session) -> Iterator[dict]:
    """Percorre todas as observações da consulta, paginando por `id_above`."""
    url = cfg["api"].rstrip("/") + "/observations"
    params = query_params(cfg)
    last_id = 0
    while True:
        page = get_json(session, url, {**params, "id_above": last_id})
        results = page.get("results", [])
        if not results:
            return
        yield from results
        last_id = results[-1]["id"]
        time.sleep(float(cfg.get("request_interval_s", 1.0)))


def fetch_observations(
    cfg: dict[str, Any], session: requests.Session, cache: Path
) -> list[dict]:
    """Baixa as observações ou reaproveita o cache `.jsonl.gz` de uma coleta anterior."""
    if cache.exists():
        with gzip.open(cache, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    from tqdm import tqdm

    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    obs = []
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for o in tqdm(iter_observations(cfg, session), desc="observações", unit="obs"):
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
            obs.append(o)
    tmp.replace(cache)
    return obs


def photo_url(url: str, size: str) -> str:
    """Troca o tamanho na URL da foto (a API devolve `square`)."""
    head, _, tail = url.rpartition("/square.")
    if not head:
        raise ValueError(f"URL de foto inesperada: {url}")
    return f"{head}/{size}.{tail}"


def eligible_photos(obs: dict, licenses: set[str], size: str) -> list[dict]:
    """Linhas (uma por foto) de uma observação, só com fotos de licença aceita.

    Descarta fotos ocultas por moderação ou com sinalização não resolvida, e observações
    cuja identificação não está no nível de espécie.
    """
    taxon = obs.get("taxon") or {}
    if taxon.get("rank") != "species":
        return []
    rows = []
    photos = sorted(obs.get("observation_photos", []), key=lambda p: p.get("position", 0))
    for pos, op in enumerate(photos):
        ph = op.get("photo") or {}
        lic = (ph.get("license_code") or "").lower()
        if lic not in licenses or ph.get("hidden"):
            continue
        if any(not fl.get("resolved", False) for fl in ph.get("flags", [])):
            continue
        rows.append(
            {
                "photo_id": ph["id"],
                "observation_id": obs["id"],
                "photo_position": pos,
                "taxon_id": taxon["id"],
                "inat_name": taxon["name"],
                "observer": (obs.get("user") or {}).get("login", ""),
                "observed_on": obs.get("observed_on") or "",
                "license": lic,
                "attribution": ph.get("attribution", ""),
                "observation_url": f"https://www.inaturalist.org/observations/{obs['id']}",
                "url": photo_url(ph["url"], size),
            }
        )
    return rows
