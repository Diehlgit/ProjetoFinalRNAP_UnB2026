"""Testes dos filtros de frequência (exigem PyTorch; rodam no Colab)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from cerrado_flora.evaluation.frequency import (
    conditions,
    high_pass,
    low_pass,
    predict_conditions,
    radial_mask,
    summarize_conditions,
)


@pytest.fixture
def x():
    g = torch.Generator().manual_seed(0)
    return torch.rand(2, 3, 32, 32, generator=g)


def test_large_radius_low_pass_is_identity(x):
    assert torch.allclose(low_pass(x, 100), x, atol=1e-5)


def test_low_plus_high_is_original(x):
    for r in (0, 2, 5, 16):
        assert torch.allclose(low_pass(x, r) + high_pass(x, r, keep_dc=False), x, atol=1e-5)


def test_low_pass_spectrum_zero_outside_radius(x):
    r = 5
    spec = torch.fft.fft2(low_pass(x, r))
    outside = ~radial_mask(32, r)
    assert spec[..., outside].abs().max() < 1e-3
    assert spec[..., ~outside].abs().max() > 1e-1


def test_high_pass_keeps_channel_mean_and_removes_low(x):
    r = 4
    y = high_pass(x, r)
    assert torch.allclose(y.mean(dim=(-2, -1)), x.mean(dim=(-2, -1)), atol=1e-5)
    spec = torch.fft.fft2(y)
    inside = radial_mask(32, r)
    inside[0, 0] = False  # DC preservado de propósito
    assert spec[..., inside].abs().max() < 1e-3


def test_outputs_real_and_same_shape(x):
    for y in (low_pass(x, 3), high_pass(x, 3)):
        assert y.shape == x.shape and y.dtype == torch.float32


def test_mask_is_symmetric():
    # simetria hermitiana: a máscara precisa conter -f sempre que contém f (saída real)
    m = radial_mask(224, 16)
    assert torch.equal(m, torch.roll(torch.flip(m, dims=(0, 1)), shifts=(1, 1), dims=(0, 1)))


def test_predict_and_summarize_conditions():
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 4))
    imgs = torch.randint(0, 256, (6, 3, 16, 16), dtype=torch.uint8)
    conds = conditions(radii=(2, 4))
    logits = predict_conditions(model, imgs, conds, torch.device("cpu"), amp=False, batch_size=4)
    assert set(logits) == {"none", "low_2", "low_4", "high_2", "high_4"}
    assert all(v.shape == (6, 4) for v in logits.values())
    labels, obs = np.array([0, 1, 2, 3, 0, 1]), np.array([1, 1, 2, 3, 4, 5])
    df = summarize_conditions(logits, labels, obs, 4)
    assert len(df) == 10 and set(df["nivel"]) == {"photo", "observation"}
    assert df.loc[df["condicao"] == "low_4", "raio"].iloc[0] == 4
