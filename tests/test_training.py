"""Testes do treino (exigem PyTorch; rodam no Colab, são pulados no ambiente local)."""

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from PIL import Image

from cerrado_flora.models.factory import build_model
from cerrado_flora.training.data import make_loaders
from cerrado_flora.training.train import lr_lambda, run_experiment


@pytest.fixture
def tiny_dataset(tmp_path):
    """3 classes; por classe: 4 observações de treino, 1 de val e 1 de teste, 2 fotos cada."""
    rng = np.random.default_rng(0)
    rows, pid, oid = [], 0, 0
    for c in range(3):
        for split in ["train"] * 4 + ["val", "test"]:
            oid += 1
            for _ in range(2):
                pid += 1
                f = f"images/{c}/{pid}.jpg"
                (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)).save(tmp_path / f)
                rows.append({"photo_id": pid, "observation_id": oid, "split": split,
                             "class_index": c, "file": f})
    pd.DataFrame(rows).to_csv(tmp_path / "images.csv", index=False)
    return tmp_path


def _cfg(epochs=2):
    return {
        "name": "teste", "model": {"arch": "gmnet_s3"},
        "data": {"img_size": 32, "resize": 36, "batch_size": 4, "num_workers": 0},
        "optim": {"lr": 1e-3, "weight_decay": 0.05},
        "schedule": {"epochs": epochs, "warmup_epochs": 1, "min_lr": 1e-6},
        "loss": {"label_smoothing": 0.1}, "train": {"amp": True, "selection_metric": "macro_f1"},
    }


def test_loaders_shapes_and_order(tiny_dataset):
    loaders = make_loaders(tiny_dataset / "images.csv", tiny_dataset, _cfg()["data"], seed=0)
    x, y = next(iter(loaders["val"]))
    assert x.shape == (4, 3, 32, 32)  # val: 6 fotos, primeiro lote de 4
    assert len(loaders["train"].dataset) == 24 and len(loaders["test"].dataset) == 6
    assert y.tolist() == loaders["val"].dataset.labels[: len(y)].tolist()


@pytest.mark.parametrize("arch", ["gmnet_s3", "starnet_s4"])
def test_model_head_has_num_classes(arch):
    m = build_model({"arch": arch}, num_classes=7, pretrained_path=None)
    assert m(torch.zeros(1, 3, 64, 64)).shape == (1, 7)


def test_starnet_s4_matches_inspected_checkpoint():
    # 7.482.728 parâmetros com cabeça de 1000 classes (inspeção do checkpoint oficial)
    m = build_model({"arch": "starnet_s4"}, num_classes=1000, pretrained_path=None)
    assert sum(p.numel() for p in m.parameters()) == 7_482_728
    assert len(m.state_dict()) == 502


def test_starnet_loader_accepts_official_checkpoint_layout(tmp_path):
    # Simula o formato do arquivo oficial: dict com state_dict, optimizer e args (Namespace)
    import argparse

    from cerrado_flora.models.starnet import load_pretrained, starnet_s4

    ref = starnet_s4()
    torch.save({"epoch": 296, "arch": "starnet_s4", "state_dict": ref.state_dict(),
                "optimizer": {}, "version": 2, "args": argparse.Namespace(lr=3e-3),
                "metric": 78.8}, tmp_path / "ck.pth.tar")
    m = load_pretrained(tmp_path / "ck.pth.tar", num_classes=30)
    assert m.head.out_features == 30
    assert torch.equal(m.stem[0].conv.weight, ref.stem[0].conv.weight)


def test_lr_schedule_warmup_and_cosine():
    f = lr_lambda(total_steps=100, warmup_steps=10, min_ratio=0.01)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(10) == pytest.approx(1.0) and f(99) == pytest.approx(0.01, abs=1e-3)


def test_run_experiment_end_to_end_and_resume(tiny_dataset, tmp_path):
    ckpt, res = tmp_path / "ckpt", tmp_path / "res"
    args = (tiny_dataset / "images.csv", tiny_dataset, None, ckpt, res)
    cfg1 = _cfg(epochs=2)
    r = run_experiment(cfg1, 0, *args)
    assert (res / "history.csv").exists() and len(pd.read_csv(res / "history.csv")) == 2
    for f in ("predictions_test.csv", "per_class_test.csv", "confusion_test.csv",
              "run_metadata.json", "metrics.json"):
        assert (res / f).exists(), f
    assert set(r["test"]) >= {"photo", "observation", "loss"}
    assert r["test"]["photo"]["n"] == 6 and r["test"]["observation"]["n"] == 3
    # Concluído: nova chamada não treina de novo
    assert run_experiment(cfg1, 0, *args) == json.loads((res / "metrics.json").read_text())


def test_resume_from_last_checkpoint(tiny_dataset, tmp_path):
    ckpt, res = tmp_path / "ckpt", tmp_path / "res"
    args = (tiny_dataset / "images.csv", tiny_dataset, None, ckpt, res)
    run_experiment(_cfg(epochs=1), 0, *args)
    (res / "metrics.json").unlink()  # simula interrupção após a época 1 de um treino de 2
    logs = []
    run_experiment(_cfg(epochs=2), 0, *args, log=logs.append)
    assert any("retomando da época 1" in m for m in logs)
    assert len(pd.read_csv(res / "history.csv")) == 2
