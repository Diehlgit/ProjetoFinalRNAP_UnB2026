"""Sensibilidade a frequências: avaliação com filtros passa-baixa e passa-alta nas imagens.

Protocolo (fixado antes de ver resultados; ver docs/decisoes.md):
- imagens de teste com o pré-processamento de avaliação (lado menor 256 + recorte central
  224), filtradas **no espaço de pixels**, antes da normalização ImageNet;
- FFT 2D por canal e máscara circular ideal de raio `r` (em índices de frequência do
  espectro 224×224; 112 é a frequência de Nyquist em cada eixo);
- passa-baixa: mantém as frequências com raio <= r;
- passa-alta: remove as frequências com raio <= r, mas **mantém a média de cada canal**
  (componente DC), para não reduzir a imagem a um quadro quase preto;
- sem recorte de valores depois do filtro (recortar reintroduziria frequências).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from cerrado_flora.evaluation.metrics import evaluate_logits
from cerrado_flora.training.data import IMAGENET_MEAN, IMAGENET_STD, load_split

DEFAULT_RADII = (4, 8, 16, 32, 64, 112)


def conditions(radii=DEFAULT_RADII) -> list[tuple[str, int | None]]:
    """Imagem original + passa-baixa e passa-alta em cada raio."""
    return [("none", None)] + [("low", r) for r in radii] + [("high", r) for r in radii]


def condition_name(kind: str, r: int | None) -> str:
    return "none" if kind == "none" else f"{kind}_{r}"


def radial_mask(size: int, r: float, device=None) -> torch.Tensor:
    """Máscara booleana (size, size) das frequências com raio <= r, na ordem de `fft2`."""
    f = torch.fft.fftfreq(size, d=1.0 / size, device=device)  # 0, 1, ..., -1 (índices)
    return (f[:, None] ** 2 + f[None, :] ** 2) <= r * r


def low_pass(x: torch.Tensor, r: float) -> torch.Tensor:
    """Passa-baixa ideal nos dois últimos eixos de `x` (imagens quadradas)."""
    size = x.shape[-1]
    assert x.shape[-2] == size, "imagens quadradas"
    spec = torch.fft.fft2(x.float())
    return torch.fft.ifft2(spec * radial_mask(size, r, x.device)).real


def high_pass(x: torch.Tensor, r: float, keep_dc: bool = True) -> torch.Tensor:
    """Complemento do passa-baixa; com `keep_dc`, a média de cada canal é preservada."""
    x = x.float()
    out = x - low_pass(x, r)
    if keep_dc:
        out = out + x.mean(dim=(-2, -1), keepdim=True)
    return out


def apply_condition(x: torch.Tensor, kind: str, r: int | None) -> torch.Tensor:
    if kind == "none":
        return x.float()
    if kind == "low":
        return low_pass(x, r)
    if kind == "high":
        return high_pass(x, r)
    raise ValueError(f"condição desconhecida: {kind}")


class _Uint8Images(Dataset):
    def __init__(self, files, root, resize, size):
        self.files, self.root = files, Path(root)
        bicubic = transforms.InterpolationMode.BICUBIC
        self.tf = transforms.Compose([
            transforms.Resize(resize, interpolation=bicubic),
            transforms.CenterCrop(size),
            transforms.PILToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image

        with Image.open(self.root / self.files[i]) as im:
            return self.tf(im.convert("RGB"))


def load_images(manifest: str | Path, root: str | Path, data_cfg: dict, split: str = "test",
                num_workers: int = 2) -> dict:
    """Decodifica uma vez as imagens da divisão (uint8, mesmo recorte da avaliação).

    Guardar em uint8 é exato: `ToTensor` produz k/255, então dividir por 255 reproduz a
    entrada usada no treino.
    """
    df = load_split(manifest, split)
    ds = _Uint8Images(df["file"].tolist(), root, int(data_cfg["resize"]), int(data_cfg["img_size"]))
    loader = DataLoader(ds, batch_size=64, num_workers=num_workers)
    images = torch.cat(list(loader))
    return {"images": images, "labels": df["class_index"].astype(int).to_numpy(),
            "observation_ids": df["observation_id"].to_numpy(),
            "photo_ids": df["photo_id"].to_numpy()}


@torch.no_grad()
def predict_conditions(model: torch.nn.Module, images: torch.Tensor, conds, device,
                       amp: bool, batch_size: int = 64) -> dict[str, np.ndarray]:
    """Logits do modelo para cada condição, na ordem das imagens."""
    model.eval()
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    out = {condition_name(k, r): [] for k, r in conds}
    for i in range(0, len(images), batch_size):
        x = images[i:i + batch_size].to(device).float() / 255.0
        for kind, r in conds:
            xf = (apply_condition(x, kind, r) - mean) / std
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out[condition_name(kind, r)].append(model(xf).float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def summarize_conditions(logits: dict[str, np.ndarray], labels: np.ndarray,
                         observation_ids: np.ndarray, n_classes: int) -> pd.DataFrame:
    """Uma linha por condição e nível (foto/observação)."""
    rows = []
    for name, lg in logits.items():
        kind, _, r = name.partition("_")
        m = evaluate_logits(lg, labels, observation_ids, n_classes)
        for level, vals in m.items():
            rows.append({"condicao": name, "tipo": kind, "raio": int(r) if r else None,
                         "nivel": level, **vals})
    return pd.DataFrame(rows)
