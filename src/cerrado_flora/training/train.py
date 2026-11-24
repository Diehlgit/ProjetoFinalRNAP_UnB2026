"""Fine-tuning e avaliação de um modelo para uma semente.

- AdamW (sem weight decay em vieses, BN e `gamma`), warmup linear + cosseno por passo,
  label smoothing, precisão mista (AMP) em GPU.
- Checkpoint `last.pt` a cada época (retomada após desconexão do Colab) e `best.pt` pelo
  macro-F1 por foto na validação.
- O teste é avaliado **uma única vez**, com o melhor checkpoint, ao final.

Saídas em `results_dir`: history.csv, metrics.json, per_class_{val,test}.csv,
confusion_test.csv, predictions_{val,test}.csv e run_metadata.json.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from cerrado_flora.data.download import sha256_file
from cerrado_flora.evaluation.metrics import (
    confusion_matrix,
    evaluate_logits,
    per_class,
    softmax,
)
from cerrado_flora.models.factory import build_model
from cerrado_flora.training.data import make_loaders, num_classes
from cerrado_flora.utils import run_metadata


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_lambda(total_steps: int, warmup_steps: int, min_ratio: float):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))
    return f


@torch.no_grad()
def predict(model: nn.Module, loader, device: torch.device, amp: bool) -> np.ndarray:
    """Logits (float32) na ordem do dataset."""
    model.eval()
    out = []
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def _eval_split(logits: np.ndarray, ds, n_classes: int) -> dict:
    m = evaluate_logits(logits, ds.labels, ds.observation_ids, n_classes)
    m["loss"] = float(F.cross_entropy(torch.from_numpy(logits),
                                      torch.from_numpy(ds.labels).long()).item())
    return m


def _predictions_frame(logits: np.ndarray, ds) -> pd.DataFrame:
    probs = softmax(logits)
    df = pd.DataFrame({"photo_id": ds.photo_ids, "observation_id": ds.observation_ids,
                       "label": ds.labels, "pred": probs.argmax(axis=1)})
    return pd.concat([df, pd.DataFrame(probs, columns=[f"p_{k}" for k in range(probs.shape[1])])],
                     axis=1)


def run_experiment(
    cfg: dict, seed: int, manifest: str | Path, images_root: str | Path,
    pretrained_path: str | Path | None, ckpt_dir: str | Path, results_dir: str | Path,
    log=print,
) -> dict:
    """Treina (ou retoma) e avalia uma semente. Se já concluída, apenas devolve as métricas."""
    ckpt_dir, results_dir = Path(ckpt_dir), Path(results_dir)
    metrics_path = results_dir / "metrics.json"
    if metrics_path.exists():
        log(f"semente {seed} já concluída: {metrics_path}")
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_metadata.set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tcfg, ocfg, scfg = cfg["train"], cfg["optim"], cfg["schedule"]
    amp = bool(tcfg.get("amp", True)) and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    n_classes = num_classes(manifest)
    loaders = make_loaders(manifest, images_root, cfg["data"], seed)
    model = build_model(cfg["model"], n_classes, pretrained_path).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, float(ocfg["weight_decay"])),
                                  lr=float(ocfg["lr"]), betas=tuple(ocfg.get("betas", (0.9, 0.999))))
    epochs = int(scfg["epochs"])
    steps_per_epoch = len(loaders["train"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda(
        epochs * steps_per_epoch, int(scfg["warmup_epochs"]) * steps_per_epoch,
        float(scfg["min_lr"]) / float(ocfg["lr"])))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["loss"].get("label_smoothing", 0.0)))
    select = tcfg.get("selection_metric", "macro_f1")
    grad_clip = tcfg.get("grad_clip")

    start_epoch, best, best_epoch, history = 0, -1.0, -1, []
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    if last_path.exists():  # arquivos gerados por este código; não são de terceiros
        # Carregado na CPU: os estados do gerador aleatório precisam ser ByteTensors de CPU;
        # model e optimizer.load_state_dict movem os tensores para o dispositivo dos parâmetros.
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        start_epoch, best, best_epoch = state["epoch"] + 1, state["best"], state["best_epoch"]
        history = state["history"]
        log(f"retomando da época {start_epoch} (melhor {select} = {best:.4f} na época {best_epoch})")

    val_ds = loaders["val"].dataset
    for epoch in range(start_epoch, epochs):
        model.train()
        t0 = time.time()
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        train_time = time.time() - t0

        val = _eval_split(predict(model, loaders["val"], device, amp), val_ds, n_classes)
        score = val["photo"][select]
        row = {
            "epoch": epoch, "lr": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / seen, "train_top1": correct / seen,
            "val_loss": val["loss"],
            **{f"val_{k}": val["photo"][k] for k in ("top1", "macro_f1", "balanced_accuracy")},
            **{f"val_obs_{k}": val["observation"][k] for k in ("top1", "macro_f1")},
            "train_time_s": train_time, "epoch_time_s": time.time() - t0,
        }
        history.append(row)
        if score > best:
            best, best_epoch = score, epoch
            torch.save(model.state_dict(), best_path)
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "epoch": epoch, "best": best, "best_epoch": best_epoch, "history": history,
        }, last_path)
        pd.DataFrame(history).to_csv(results_dir / "history.csv", index=False)
        log(f"época {epoch + 1}/{epochs}  loss {row['train_loss']:.3f}  "
            f"val top1 {row['val_top1']:.3f}  val macro-F1 {row['val_macro_f1']:.3f}  "
            f"({row['epoch_time_s']:.0f}s)")

    # Avaliação final com o melhor checkpoint: validação e, uma única vez, teste.
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    results = {"seed": seed, "best_epoch": best_epoch, "selection_metric": f"val photo {select}"}
    for split in ("val", "test"):
        ds = loaders[split].dataset
        logits = predict(model, loaders[split], device, amp)
        results[split] = _eval_split(logits, ds, n_classes)
        _predictions_frame(logits, ds).to_csv(results_dir / f"predictions_{split}.csv", index=False)
        cm = confusion_matrix(ds.labels, logits.argmax(axis=1), n_classes)
        per_class(cm).rename_axis("class_index").to_csv(results_dir / f"per_class_{split}.csv")
        if split == "test":
            pd.DataFrame(cm).to_csv(results_dir / "confusion_test.csv")

    run_metadata.save(results_dir / "run_metadata.json", extra={
        "experiment": cfg["name"], "seed": seed, "config": cfg, "device": str(device),
        "manifest_sha256": sha256_file(Path(manifest)),
        "pretrained_sha256": sha256_file(Path(pretrained_path)) if pretrained_path else None,
        "n_classes": n_classes,
        "n_images": {s: len(loaders[s].dataset) for s in loaders},
    })
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"teste (foto): {results['test']['photo']}")
    return results
