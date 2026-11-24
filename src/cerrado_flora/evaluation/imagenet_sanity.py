"""Teste de sanidade de pesos ImageNet-1K em uma amostra de imagens do ImageNet.

Protocolo (o mesmo usado para o GmNet-S3 em 2026-11-23, ver docs/decisoes.md):
repositório `EliSchwartz/imagenet-sample-images` (1 imagem por classe), uma classe a cada 5
(200 imagens); redimensionar o lado menor para 256 (bicúbico), recorte central 224,
normalização ImageNet. O índice da classe é a posição do *wnid* na lista ordenada.

Não substitui a avaliação no conjunto de validação do ImageNet: só confirma que os pesos
foram carregados na arquitetura certa (pesos errados dão ~0%).
"""

from __future__ import annotations

from pathlib import Path

import requests
import torch

REPO = "EliSchwartz/imagenet-sample-images"
TREE_URL = f"https://api.github.com/repos/{REPO}/git/trees/master"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/master/{{}}"


def download_sample(dest: str | Path, every: int = 5) -> list[tuple[Path, int]]:
    """Baixa (se preciso) uma imagem a cada `every` classes; retorna (arquivo, classe)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tree = requests.get(TREE_URL, timeout=60).json()["tree"]
    names = sorted(t["path"] for t in tree if t["path"].endswith(".JPEG"))
    if len(names) != 1000:
        raise RuntimeError(f"esperadas 1000 imagens, encontradas {len(names)}")
    out = []
    for idx in range(0, 1000, every):
        f = dest / names[idx]
        if not f.exists():
            r = requests.get(RAW_URL.format(names[idx]), timeout=60)
            r.raise_for_status()
            f.write_bytes(r.content)
        out.append((f, idx))
    return out


@torch.no_grad()
def imagenet_topk(model: torch.nn.Module, samples: list[tuple[Path, int]],
                  device: torch.device) -> dict[str, float]:
    from PIL import Image

    from cerrado_flora.training.data import build_transforms

    tf = build_transforms({"img_size": 224, "resize": 256}, train=False)
    model.eval().to(device)
    top1 = top5 = 0
    for f, label in samples:
        with Image.open(f) as im:
            x = tf(im.convert("RGB")).unsqueeze(0).to(device)
        pred = model(x).topk(5, dim=1).indices[0].tolist()
        top1 += pred[0] == label
        top5 += label in pred
    n = len(samples)
    return {"top1": top1 / n, "top5": top5 / n, "n": n}
