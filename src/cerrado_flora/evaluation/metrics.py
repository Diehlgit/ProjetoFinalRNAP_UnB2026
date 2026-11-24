"""Métricas de classificação (só NumPy, para testar sem PyTorch).

Métricas reportadas (decisão de 2026-11-24): acurácia top-1, macro-F1 e acurácia
balanceada, por foto e por observação. A seleção do checkpoint usa o macro-F1 por foto
na validação.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Linhas = classe verdadeira, colunas = classe predita."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def per_class(cm: np.ndarray) -> pd.DataFrame:
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return pd.DataFrame({"precision": precision, "recall": recall, "f1": f1,
                         "support": support})


def summarize(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    """Top-1, macro-F1 e acurácia balanceada (médias sobre as classes presentes)."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    pc = per_class(cm)
    present = pc["support"] > 0
    return {
        "top1": float(np.trace(cm) / cm.sum()),
        "macro_f1": float(pc.loc[present, "f1"].mean()),
        "balanced_accuracy": float(pc.loc[present, "recall"].mean()),
        "n": int(cm.sum()),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def aggregate_by_observation(
    probs: np.ndarray, observation_ids: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Média das probabilidades das fotos de cada observação.

    Retorna (ids das observações, rótulos, probabilidades médias), em ordem de id.
    """
    obs_ids, inverse = np.unique(observation_ids, return_inverse=True)
    sums = np.zeros((len(obs_ids), probs.shape[1]))
    np.add.at(sums, inverse, probs)
    counts = np.bincount(inverse, minlength=len(obs_ids))[:, None]
    obs_labels = np.zeros(len(obs_ids), dtype=np.int64)
    obs_labels[inverse] = labels
    if not np.all(obs_labels[inverse] == labels):
        raise ValueError("fotos da mesma observação com rótulos diferentes")
    return obs_ids, obs_labels, sums / counts


def evaluate_logits(
    logits: np.ndarray, labels: np.ndarray, observation_ids: np.ndarray, n_classes: int
) -> dict[str, dict[str, float]]:
    """Métricas por foto e por observação a partir dos logits."""
    probs = softmax(logits)
    _, obs_labels, obs_probs = aggregate_by_observation(probs, observation_ids, labels)
    return {
        "photo": summarize(labels, probs.argmax(axis=1), n_classes),
        "observation": summarize(obs_labels, obs_probs.argmax(axis=1), n_classes),
    }
