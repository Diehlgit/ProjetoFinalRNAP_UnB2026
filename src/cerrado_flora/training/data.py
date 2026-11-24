"""Dataset a partir do manifesto (`data/manifests/images.csv`) e transformações.

Treino: recorte aleatório redimensionado + espelhamento horizontal (sem alteração de cor,
que é informativa para flores e folhas). Validação/teste: redimensionar o lado menor para
`resize` (bicúbico) + recorte central `img_size`, como no teste de sanidade dos pesos.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(data_cfg: dict, train: bool) -> transforms.Compose:
    size = int(data_cfg["img_size"])
    bicubic = transforms.InterpolationMode.BICUBIC
    norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if train:
        scale = tuple(data_cfg.get("train_crop_scale", (0.25, 1.0)))
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=scale, interpolation=bicubic),
            transforms.RandomHorizontalFlip(),
            *norm,
        ])
    return transforms.Compose([
        transforms.Resize(int(data_cfg["resize"]), interpolation=bicubic),
        transforms.CenterCrop(size),
        *norm,
    ])


class ManifestDataset(Dataset):
    """Fotos de uma divisão do manifesto, na ordem do manifesto."""

    def __init__(self, images: pd.DataFrame, root: str | Path, transform):
        self.root = Path(root)
        self.files = images["file"].tolist()
        self.labels = images["class_index"].astype(int).to_numpy()
        self.photo_ids = images["photo_id"].to_numpy()
        self.observation_ids = images["observation_id"].to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        with Image.open(self.root / self.files[i]) as im:
            x = self.transform(im.convert("RGB"))
        return x, int(self.labels[i])


def load_split(manifest: str | Path, split: str) -> pd.DataFrame:
    images = pd.read_csv(manifest)
    return images[images["split"] == split].reset_index(drop=True)


def num_classes(manifest: str | Path) -> int:
    return int(pd.read_csv(manifest, usecols=["class_index"])["class_index"].max()) + 1


def make_loaders(manifest: str | Path, root: str | Path, data_cfg: dict, seed: int,
                 splits=("train", "val", "test")) -> dict[str, DataLoader]:
    """DataLoaders por divisão; só o de treino embaralha (com gerador semeado)."""
    loaders = {}
    workers = int(data_cfg.get("num_workers", 2))
    for split in splits:
        train = split == "train"
        ds = ManifestDataset(load_split(manifest, split), root, build_transforms(data_cfg, train))
        g = torch.Generator()
        g.manual_seed(seed)
        loaders[split] = DataLoader(
            ds, batch_size=int(data_cfg["batch_size"]), shuffle=train, drop_last=train,
            num_workers=workers, pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0, generator=g,
        )
    return loaders
