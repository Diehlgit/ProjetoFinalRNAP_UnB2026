"""GmNet (WANG et al., 2026, ICLR 2026).

Arquitetura copiada do código oficial (https://github.com/YFWang1999/GmNet, arquivo
`gment.py`, consultado em 2026-11-23), que não declara licença. Uso exclusivamente
acadêmico, com citação. Alterações em relação ao original:
- imports não utilizados removidos; `timm.models.layers` substituído por `timm.layers`;
- removidos `register_model`, `PermuteLienar` e `convert_model` (não usados aqui);
- `gmnet_s3` com `f12_bn=True` e `dwconv2_bn=True`, a configuração que corresponde ao
  checkpoint público `gmnet_s3.npy` (o código oficial usa `False`; ver docs/decisoes.md);
- adicionada `load_pretrained`, que carrega o checkpoint `.npy` com segurança;
- adicionado o parâmetro `gating` (padrão `True`, comportamento original). Com
  `gating=False` o bloco calcula `act(x)` em vez de `act(x) * x`; é a única diferença,
  e o gating não tem parâmetros, então as duas variantes têm os mesmos pesos (ablação
  do gating, ver docs/decisoes.md).
"""
# Estilo do código original mantido de propósito, para facilitar a comparação.
# ruff: noqa: B006, PLR0402, UP034

import io
import pickle

import numpy as np
import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, kernel_size=7,
                 f12_bn=False, g_bn=True, dwconv2_bn=False, act=nn.ReLU,
                 drop_path=0., layer_scale=1e-6, gating=True):
        super().__init__()
        self.gating = gating
        self.dwconv = ConvBN(dim, dim, kernel_size, 1, (kernel_size - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=f12_bn)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=g_bn)
        self.dwconv2 = ConvBN(dim, dim, kernel_size, 1, (kernel_size - 1) // 2, groups=dim,
                              with_bn=dwconv2_bn)
        self.act = act()
        self.gamma = nn.Parameter(layer_scale * torch.ones((dim)),
                                  requires_grad=True) if layer_scale > 0 else None
        if drop_path > 0.:
            self.drop_path = DropPath(drop_path)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.f1(x)
        if self.gating:
            x = self.act(x) * (x)  # mecanismo de gating
        else:
            x = self.act(x)  # ablação: só a ativação, sem a multiplicação
        x = self.g(x)
        x = self.dwconv2(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        if hasattr(self, "drop_path"):
            x = input + self.drop_path(x)
        else:
            x = input + x
        return x


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1,
                 groups=1, with_bn=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.add_module('conv', torch.nn.Conv2d(
            in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Model(nn.Module):
    def __init__(self, num_classes=1000,
                 embed_dim=[32, 64, 128, 256], depths=[2, 2, 8, 2],
                 f12_bn=False, g_bn=False, dwconv2_bn=False, act=nn.ReLU, downsampler_act=nn.ReLU,
                 mlp_ratio=[4, 4, 4, 4], layer_scale=1e-6,
                 drop_path_rate=0.0, kernel_size=7, block=None, gating=True,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32

        self.stem = nn.Sequential(
            ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1),
            act(),
        )
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            down_sampler = nn.Sequential(
                ConvBN(self.in_channel, embed_dim[i_layer], 3, 2, 1),
                downsampler_act()
            )
            self.in_channel = embed_dim[i_layer]
            blocks = [
                block(self.in_channel, mlp_ratio=mlp_ratio[i_layer], kernel_size=kernel_size,
                      f12_bn=f12_bn, g_bn=g_bn, dwconv2_bn=dwconv2_bn, act=act,
                      drop_path=dpr[cur + i], layer_scale=layer_scale, gating=gating)
                for i in range(depths[i_layer])]
            cur += depths[i_layer]
            stage = nn.Sequential(down_sampler, *blocks)
            self.stages.append(stage)
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
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def gmnet_s3(**kwargs):
    return Model(
        embed_dim=[48, 96, 192, 384], depths=[3, 3, 8, 3], mlp_ratio=[4, 4, 4, 4], kernel_size=7,
        f12_bn=True, g_bn=True, dwconv2_bn=True, act=nn.ReLU6, downsampler_act=nn.Identity,
        block=Block, **kwargs,
    )


class _NumpyOnlyUnpickler(pickle.Unpickler):
    """Só aceita os objetos NumPy presentes no checkpoint; recusa qualquer outro."""

    def find_class(self, module, name):
        if (module, name) in {("numpy.core.multiarray", "_reconstruct"),
                              ("numpy._core.multiarray", "_reconstruct")}:
            core = getattr(np, "_core", None) or np.core  # NumPy 2.x / 1.x
            return core.multiarray._reconstruct
        if module == "numpy" and name in ("ndarray", "dtype"):
            return getattr(np, name)
        raise pickle.UnpicklingError(f"Objeto não permitido no checkpoint: {module}.{name}")


def load_pretrained(path, num_classes=1000, **kwargs):
    """Cria o GmNet-S3 com os pesos ImageNet-1K de `path` (gmnet_s3.npy).

    Se `num_classes` for diferente de 1000, a camada `head` é substituída por uma nova,
    inicializada aleatoriamente, para o fine-tuning.
    """
    with open(path, "rb") as f:
        data = f.read()
    header_end = data.index(b"\n", 10) + 1  # fim do cabeçalho do formato .npy
    obj = _NumpyOnlyUnpickler(io.BytesIO(data[header_end:])).load()
    state_dict = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in obj.item().items()}

    model = gmnet_s3(**kwargs)
    model.load_state_dict(state_dict, strict=True)
    if num_classes != 1000:
        model.head = nn.Linear(model.head.in_features, num_classes)
        trunc_normal_(model.head.weight, std=.02)
        nn.init.constant_(model.head.bias, 0)
    return model
