"""Linha de comando do dataset.

Construir (consulta + validação + divisão; gera os manifestos):
    python -m cerrado_flora.data build --config configs/dataset.yaml --out outputs/dataset --zip

Recriar a partir dos manifestos versionados (mesmas fotos, checksums verificados):
    python -m cerrado_flora.data fetch --out outputs/dataset --zip
"""

from __future__ import annotations

import argparse
import json

from cerrado_flora.data.dataset import build, fetch


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m cerrado_flora.data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="constrói o dataset a partir do config")
    b.add_argument("--config", default="configs/dataset.yaml")
    f = sub.add_parser("fetch", help="recria as imagens a partir dos manifestos")
    f.add_argument("--workers", type=int, default=4)
    for p in (b, f):
        p.add_argument("--out", default="outputs/dataset", help="pasta local das imagens")
        p.add_argument("--manifests", default="data/manifests")
        p.add_argument("--zip", action="store_true", help="gera <nome>.zip para o Drive")
    a = ap.parse_args()
    if a.cmd == "build":
        info = build(a.config, a.out, a.manifests, make_zip=a.zip)
        print(json.dumps({k: info[k] for k in ("funil", "por_divisao")}, indent=2,
                         ensure_ascii=False))
    else:
        rep = fetch(a.manifests, a.out, a.workers, make_zip=a.zip)
        print(json.dumps({k: v for k, v in rep.items() if not isinstance(v, list)}, indent=2,
                         ensure_ascii=False))
        print(f"ausentes: {len(rep['ausentes'])}, divergentes: {len(rep['checksum_divergente'])}")


if __name__ == "__main__":
    main()
