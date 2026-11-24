"""StarNet (MA et al., 2024, CVPR 2024: "Rewrite the Stars") — baseline do projeto.

Arquitetura copiada do código oficial (https://github.com/ma-xu/Rewrite-the-Stars, arquivo
`imagenet/starnet.py`, consultado em 2026-11-24), sob licença Apache-2.0
(Copyright Xu Ma). Alterações em relação ao original:
- `timm.models.layers` substituído por `timm.layers`; `register_model` e `model_urls`
  removidos (URL e checksum dos pesos ficam em `configs/pretrained.yaml`);
- mantida apenas a variante `starnet_s4` (removidas s1–s3 e s050–s150);
- `starnet_s4` sem o argumento `pretrained`; adicionada `load_pretrained`, que carrega
  o checkpoint oficial com `torch.load(weights_only=True)`.

Diferença central em relação ao GmNet (ver docs/decisoes.md): o bloco da StarNet
multiplica dois ramos, `ReLU6(f1(x)) * f2(x)`; o do GmNet, `ReLU6(f1(x)) * f1(x)`.
"""
# Estilo do código original mantido de propósito, para facilitar a comparação.
# ruff: noqa: B006, PLR0402

import argparse

import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2  # operação "star" (dois ramos)
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Mantido como no original (note que `nn.Linear or nn.Conv2d` avalia para nn.Linear).
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)


def starnet_s4(**kwargs):
    return StarNet(32, [3, 3, 12, 5], **kwargs)


def load_pretrained(path, num_classes=1000, **kwargs):
    """Cria a StarNet-S4 com os pesos ImageNet-1K do checkpoint oficial (`starnet_s4.pth.tar`).

    O arquivo contém `state_dict`, `optimizer`, `args` (um `argparse.Namespace`), `epoch`,
    `metric` e `version`. Carregado com `weights_only=True`, liberando só o
    `argparse.Namespace`; apenas o `state_dict` é usado. Se `num_classes` for diferente de
    1000, a camada `head` é substituída por uma nova, inicializada aleatoriamente.
    """
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = starnet_s4(**kwargs)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if num_classes != 1000:
        model.head = nn.Linear(model.head.in_features, num_classes)
        trunc_normal_(model.head.weight, std=.02)
        nn.init.constant_(model.head.bias, 0)
    return model
