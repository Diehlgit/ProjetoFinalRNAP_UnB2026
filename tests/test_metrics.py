"""Testes das métricas (NumPy puro; rodam também sem PyTorch)."""

import numpy as np
import pytest

from cerrado_flora.evaluation.metrics import (
    aggregate_by_observation,
    confusion_matrix,
    evaluate_logits,
    summarize,
)


def test_summarize_known_values():
    # classe 0: 4 exemplos, 3 corretos; classe 1: 2 exemplos, 1 correto (1 predito como 0)
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0, 0, 0, 1, 1, 0])
    s = summarize(y, p, 2)
    assert s["top1"] == pytest.approx(4 / 6)
    assert s["balanced_accuracy"] == pytest.approx((3 / 4 + 1 / 2) / 2)
    f1_0 = 2 * (3 / 4) * (3 / 4) / (3 / 4 + 3 / 4)
    f1_1 = 2 * (1 / 2) * (1 / 2) / 1
    assert s["macro_f1"] == pytest.approx((f1_0 + f1_1) / 2)


def test_class_never_predicted_counts_as_zero_f1():
    s = summarize(np.array([0, 1, 2]), np.array([0, 0, 2]), 3)
    assert s["macro_f1"] == pytest.approx((2 / 3 + 0 + 1) / 3)


def test_confusion_matrix_orientation():
    cm = confusion_matrix(np.array([0, 1]), np.array([1, 1]), 2)
    assert cm.tolist() == [[0, 1], [0, 1]]


def test_aggregate_by_observation_averages_probabilities():
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]])
    ids, labels, mean = aggregate_by_observation(probs, np.array([7, 7, 7, 3]),
                                                 np.array([0, 0, 0, 1]))
    assert ids.tolist() == [3, 7] and labels.tolist() == [1, 0]
    assert mean[1] == pytest.approx([1.4 / 3, 1.6 / 3])
    with pytest.raises(ValueError):
        aggregate_by_observation(probs, np.array([7, 7, 7, 3]), np.array([0, 1, 0, 1]))


def test_evaluate_logits_photo_vs_observation():
    # observação 1: duas fotos erradas por pouco e uma certa com folga -> certa na média
    logits = np.array([[0.0, 0.2], [0.0, 0.2], [3.0, 0.0], [0.0, 2.0]])
    r = evaluate_logits(logits, np.array([0, 0, 0, 1]), np.array([1, 1, 1, 2]), 2)
    assert r["photo"]["top1"] == pytest.approx(0.5)
    assert r["observation"]["top1"] == pytest.approx(1.0)
