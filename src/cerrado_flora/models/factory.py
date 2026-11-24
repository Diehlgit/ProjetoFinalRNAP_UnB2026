"""Construção dos modelos a partir da configuração do experimento.

Ponto único de criação dos modelos, para que baseline e GmNet passem pelo mesmo
código de treino (regra 7: mesmas condições experimentais).
"""

from __future__ import annotations

from pathlib import Path

from torch import nn

from cerrado_flora.data.download import sha256_file


def verify_checkpoint(path: str | Path, expected_sha256: str) -> None:
    """Confere se o arquivo de pesos é a cópia verificada (configs/pretrained.yaml)."""
    digest = sha256_file(Path(path))
    if digest != expected_sha256:
        raise ValueError(f"SHA-256 de {path} diverge do esperado: {digest}")


def build_model(model_cfg: dict, num_classes: int, pretrained_path: str | Path | None) -> nn.Module:
    """Cria o modelo. `pretrained_path=None` cria com pesos aleatórios (treino do zero).

    `model_cfg["gating"]` (só GmNet, padrão `True`) desliga o gating na ablação.
    """
    arch = model_cfg["arch"]
    kwargs = {"drop_path_rate": float(model_cfg.get("drop_path_rate", 0.0))}
    gating = bool(model_cfg.get("gating", True))
    if arch != "gmnet_s3" and not gating:
        raise ValueError(f"gating=False só é suportado no gmnet_s3, não em {arch}")
    if arch == "gmnet_s3":
        kwargs["gating"] = gating
        from cerrado_flora.models.gmnet import gmnet_s3, load_pretrained

        if pretrained_path is None:
            return gmnet_s3(num_classes=num_classes, **kwargs)
        return load_pretrained(pretrained_path, num_classes=num_classes, **kwargs)
    if arch == "starnet_s4":
        from cerrado_flora.models.starnet import load_pretrained, starnet_s4

        if pretrained_path is None:
            return starnet_s4(num_classes=num_classes, **kwargs)
        return load_pretrained(pretrained_path, num_classes=num_classes, **kwargs)
    raise ValueError(f"arquitetura desconhecida: {arch}")
