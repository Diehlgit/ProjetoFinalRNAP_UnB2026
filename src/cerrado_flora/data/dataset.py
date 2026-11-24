"""Construção (`build`) e recriação (`fetch`) do dataset a partir de `configs/dataset.yaml`.

- `build`: consulta o iNaturalist, valida as espécies na Flora e Funga do Brasil, seleciona
  e limita as classes, baixa as fotos, agrupa quase-duplicatas, divide por classe e escreve
  os manifestos em `data/manifests/` (versionados no Git).
- `fetch`: recria exatamente o mesmo conjunto a partir de `data/manifests/images.csv`,
  verificando o SHA-256 de cada foto. Não refaz consulta nem divisão.

O iNaturalist muda com o tempo (identificações, licenças, fotos removidas); por isso o
dataset reportado é definido pelo manifesto, e não pela consulta.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from cerrado_flora.data import inaturalist as inat
from cerrado_flora.data.download import download, download_many, sha256_file
from cerrado_flora.data.flora_brasil import FloraBrasil
from cerrado_flora.data.split import (
    SPLITS,
    cap_observations,
    check_leakage,
    class_rng,
    dhash,
    group_observations,
    near_duplicate_pairs,
    observer_overlap,
    split_class,
)

DEFAULT_USER_AGENT = "cerrado-flora-dataset/0.1 (academic research)"

IMAGE_COLUMNS = [
    "photo_id", "observation_id", "group_id", "split", "class_index", "scientific_name",
    "taxon_id", "photo_position", "observer", "observed_on", "license", "attribution",
    "observation_url", "url", "file", "sha256", "dhash", "width", "height",
]


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ext(url: str) -> str:
    return url.rsplit(".", 1)[-1].lower()


def _get_flora(native: dict, session, work: Path) -> Path:
    path = work / f"flora_brasil_{native['version']}.zip"
    err = download(session, native["dwca_url"], path)
    if err:
        raise RuntimeError(f"falha ao baixar a Flora e Funga do Brasil: {err}")
    digest = sha256_file(path)
    if native.get("sha256") and digest != native["sha256"]:
        raise RuntimeError(f"SHA-256 da Flora e Funga diverge do config: {digest}")
    return path


def _species_table(photos: pd.DataFrame, cfg: dict, flora: FloraBrasil) -> pd.DataFrame:
    native, sel = cfg["native"], cfg["selection"]
    counts = (photos.groupby(["taxon_id", "inat_name"])["observation_id"].nunique()
              .reset_index(name="n_obs_elegiveis"))
    cand = counts[counts["n_obs_elegiveis"] >= sel["min_observations"]].copy()
    checks = {n: flora.check(n, native.get("manual")) for n in cand["inat_name"]}
    sp = cand.merge(pd.DataFrame([c.as_row() for c in checks.values()]), on="inat_name")
    sp["nativa_cerrado"] = [
        checks[n].is_native(native["establishment"], native["domain"]) for n in sp["inat_name"]
    ]
    status = []
    for _, r in sp.iterrows():
        if r["match"] in ("ambiguo", "nao_encontrado"):
            status.append("revisao_manual")
        elif not r["domains"] or not r["establishment"]:
            # Ausência de informação na fonte não é evidência de não ser nativa do Cerrado.
            status.append("revisao_manual")
        elif not r["nativa_cerrado"]:
            status.append("excluida_criterio_nativa")
        else:
            status.append("incluida")
    sp["status"] = status
    # Dois táxons do iNaturalist que caem no mesmo nome aceito não podem ser duas classes.
    inc = sp["status"] == "incluida"
    dup = inc & sp["ffb_id"].duplicated(keep=False) & sp["ffb_id"].ne("")
    sp.loc[dup, "status"] = "revisao_manual"
    sp.loc[dup, "note"] = sp.loc[dup, "note"] + " | mesmo nome aceito que outro táxon"
    return sp.rename(columns={"inat_name": "scientific_name"}).sort_values("scientific_name")


def build(config_path: str | Path, out_dir: str | Path, manifest_dir: str | Path,
          make_zip: bool = False) -> dict:
    cfg = load_config(config_path)
    icfg, sel, scfg = cfg["inaturalist"], cfg["selection"], cfg["split"]
    seed = int(scfg["seed"])
    out, mdir = Path(out_dir), Path(manifest_dir)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    session = inat.make_session(icfg.get("user_agent", DEFAULT_USER_AGENT))
    info: dict[str, Any] = {"dataset": cfg["name"], "config": cfg, "build_started_utc": _now()}

    # 1. Observações e fotos elegíveis (licença verificada foto a foto).
    cache = work / f"inat_observations_created_d2_{icfg['created_before']}.jsonl.gz"
    qmeta = work / "inat_query.json"
    if not cache.exists():
        qmeta.write_text(json.dumps({"query_utc": _now(), "params": inat.query_params(icfg)}))
    obs = inat.fetch_observations(icfg, session, cache)
    info["inat_query"] = json.loads(qmeta.read_text()) if qmeta.exists() else None
    licenses = {lic.lower() for lic in icfg["photo_licenses"]}
    photos = pd.DataFrame(
        [r for o in obs for r in inat.eligible_photos(o, licenses, icfg["photo_size"])]
    )
    funnel = {
        "observacoes_retornadas_api": len(obs),
        "observacoes_especie_com_foto_elegivel": int(photos["observation_id"].nunique()),
        "fotos_elegiveis": len(photos),
        "especies": int(photos["taxon_id"].nunique()),
    }

    # 2. Espécies candidatas e validação de nativas.
    flora_zip = _get_flora(cfg["native"], session, work)
    info["flora_brasil_sha256"] = sha256_file(flora_zip)
    species = _species_table(photos, cfg, FloraBrasil.from_dwca(flora_zip))
    funnel["especies_com_min_observacoes"] = len(species)
    funnel["especies_nativas_cerrado"] = int((species["status"] == "incluida").sum())
    included = species.loc[species["status"] == "incluida", "taxon_id"].tolist()

    # 3. Limite de observações por classe (sorteio com semente por classe).
    kept_obs = []
    for tid in included:
        ids = photos.loc[photos["taxon_id"] == tid, "observation_id"].unique().tolist()
        kept_obs += cap_observations(ids, sel["max_observations"], class_rng(seed, tid))
    sel_photos = photos[photos["observation_id"].isin(kept_obs)].copy()
    sel_photos["file"] = [
        f"images/{t}/{p}.{_ext(u)}"
        for t, p, u in zip(sel_photos["taxon_id"], sel_photos["photo_id"], sel_photos["url"])
    ]

    # 4. Download, checksum e hash perceptual.
    jobs = [(u, out / f) for u, f in zip(sel_photos["url"], sel_photos["file"])]
    failures = download_many(session, jobs, int(cfg["download"]["workers"]))
    failed_files = {str(p.relative_to(out)) for p in failures}
    dropped = [{"photo_id": int(r.photo_id), "motivo": f"download: {failures[out / r.file]}"}
               for r in sel_photos.itertuples() if r.file in failed_files]
    sel_photos = sel_photos[~sel_photos["file"].isin(failed_files)].reset_index(drop=True)
    from tqdm import tqdm

    meta = []
    for f in tqdm(sel_photos["file"], desc="hashes", unit="foto"):
        from PIL import Image

        p = out / f
        with Image.open(p) as im:
            w, h = im.size
        meta.append((sha256_file(p), dhash(p), w, h))
    sel_photos["sha256"], hashes, sel_photos["width"], sel_photos["height"] = map(list, zip(*meta))
    sel_photos["dhash"] = [f"{x:016x}" for x in hashes]

    # 5. Quase-duplicatas: mesma classe → mesmo grupo; classes diferentes → descartadas.
    pairs = near_duplicate_pairs(hashes, int(cfg["dedup"]["dhash_max_distance"]))
    links, cross = [], set()
    for i, j, dist in pairs:
        a, b = sel_photos.iloc[i], sel_photos.iloc[j]
        if a["observation_id"] == b["observation_id"]:
            continue
        if a["taxon_id"] == b["taxon_id"]:
            links.append((int(a["observation_id"]), int(b["observation_id"])))
        else:
            cross.update([i, j])
    for i in sorted(cross):
        dropped.append({"photo_id": int(sel_photos.iloc[i]["photo_id"]),
                        "motivo": "quase-duplicata em outra classe"})
    sel_photos = sel_photos.drop(index=sorted(cross))
    exact = sel_photos.sort_values("photo_id").duplicated("sha256")
    for pid in sel_photos.loc[exact[exact].index, "photo_id"]:
        dropped.append({"photo_id": int(pid), "motivo": "cópia exata de outra foto"})
    # As cópias exatas também ligam observações (já estão em `links` via dHash = 0).
    sel_photos = sel_photos.drop(index=exact[exact].index)
    remaining = set(sel_photos["observation_id"].astype(int))
    links = [(a, b) for a, b in links if a in remaining and b in remaining]
    groups = group_observations(sorted(remaining), links)
    sel_photos["group_id"] = sel_photos["observation_id"].map(groups)

    # 6. Classes que ainda têm o mínimo; índice em ordem alfabética do nome.
    n_obs = sel_photos.groupby("taxon_id")["observation_id"].nunique()
    short = n_obs[n_obs < sel["min_observations"]].index
    species.loc[species["taxon_id"].isin(short), "status"] = "excluida_min_apos_download"
    sel_photos = sel_photos[~sel_photos["taxon_id"].isin(short)]
    final = species[species["status"] == "incluida"].sort_values("scientific_name")
    class_index = {t: k for k, t in enumerate(final["taxon_id"])}
    species["class_index"] = species["taxon_id"].map(class_index).astype("Int64")
    species["n_obs_usadas"] = species["taxon_id"].map(n_obs).astype("Int64")
    species.loc[species["class_index"].isna(), "n_obs_usadas"] = pd.NA

    # 7. Divisão classe a classe, por grupo de observações.
    assign: dict[int, str] = {}
    for tid in final["taxon_id"]:
        d = sel_photos[sel_photos["taxon_id"] == tid].drop_duplicates("observation_id")
        g = dict(zip(d["observation_id"].astype(int), d["group_id"].astype(int)))
        assign.update(split_class(g, scfg["proportions"], class_rng(seed + 1, tid)))
    sel_photos["split"] = sel_photos["observation_id"].map(assign)
    sel_photos["class_index"] = sel_photos["taxon_id"].map(class_index)
    sel_photos = sel_photos.rename(columns={"inat_name": "scientific_name"})
    images = sel_photos[IMAGE_COLUMNS].sort_values(["class_index", "observation_id", "photo_position"])
    check_leakage(images)

    # 8. Manifestos e resumo.
    obs_split = images.drop_duplicates("observation_id")
    for s in SPLITS:
        col = f"n_obs_{s}"
        species[col] = species["taxon_id"].map(
            obs_split[obs_split["split"] == s].groupby("taxon_id").size()).astype("Int64")
    overlap = observer_overlap(images)
    info.update({
        "build_finished_utc": _now(),
        "funil": funnel | {
            "classes_finais": len(final),
            "observacoes_finais": int(images["observation_id"].nunique()),
            "fotos_finais": len(images),
        },
        "por_divisao": {
            s: {"observacoes": int((obs_split["split"] == s).sum()),
                "fotos": int((images["split"] == s).sum())} for s in SPLITS
        },
        "grupos_com_mais_de_uma_observacao": int(
            (obs_split.groupby("group_id").size() > 1).sum()),
        "sobreposicao_observador_media": {
            s: float(overlap[f"{s}_obs_observer_in_train"].mean()) for s in ("val", "test")
        },
        "fotos_descartadas": dropped,
        "codigo": _code_info(),
    })
    mdir.mkdir(parents=True, exist_ok=True)
    images.to_csv(mdir / "images.csv", index=False)
    species.to_csv(mdir / "species.csv", index=False)
    overlap.to_csv(mdir / "observer_overlap.csv", index=False)
    (mdir / "build_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    if make_zip:
        info["zip"] = str(pack(out, mdir, cfg["name"]))
    return info


def _code_info() -> dict:
    from cerrado_flora.utils.run_metadata import collect

    meta = collect()
    meta.pop("pip_freeze", None)
    import numpy
    import PIL

    meta.update({"pandas": pd.__version__, "numpy": numpy.__version__, "pillow": PIL.__version__})
    return meta


def fetch(manifest_dir: str | Path, out_dir: str | Path, workers: int = 4,
          make_zip: bool = False, user_agent: str = DEFAULT_USER_AGENT) -> dict:
    """Recria as imagens listadas em `images.csv` e verifica os checksums."""
    mdir, out = Path(manifest_dir), Path(out_dir)
    images = pd.read_csv(mdir / "images.csv")
    session = inat.make_session(user_agent)
    jobs = [(u, out / f) for u, f in zip(images["url"], images["file"])]
    failures = download_many(session, jobs, workers)
    mismatched = []
    for f, expected in zip(images["file"], images["sha256"]):
        p = out / f
        if p.exists() and sha256_file(p) != expected:
            mismatched.append(f)
    report = {
        "fetch_utc": _now(),
        "fotos_no_manifesto": len(images),
        "ausentes": sorted(str(p.relative_to(out)) for p in failures),
        "checksum_divergente": mismatched,
    }
    report["completo"] = not report["ausentes"] and not mismatched
    (out / "fetch_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if make_zip:
        name = json.loads((mdir / "build_info.json").read_text(encoding="utf-8"))["dataset"]
        report["zip"] = str(pack(out, mdir, name))
    return report


def pack(out: Path, mdir: Path, name: str) -> Path:
    """Empacota imagens + manifestos num zip (sem recompressão) para enviar ao Drive."""
    images = pd.read_csv(mdir / "images.csv")
    dest = out / f"{name}.zip"
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_STORED) as z:
        for f in images["file"]:
            z.write(out / f, f"{name}/{f}")
        for m in ("images.csv", "species.csv", "observer_overlap.csv", "build_info.json"):
            z.write(mdir / m, f"{name}/manifests/{m}")
    return dest
