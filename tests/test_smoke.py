"""Pytest smoke: run two representative experiments and assert real metrics.

Offline, CPU only, sklearn built-in data. No downloads, no network.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from smoke import breast_cancer_experiment, iris_experiment  # noqa: E402


def test_iris_experiment_produces_metrics():
    out = iris_experiment()
    assert out["n_samples"] == 150
    # Four models each produced a held out accuracy and a CV mean.
    assert set(out["holdout"]) == {"LR", "DT", "RF", "SVM"}
    assert set(out["cv"]) == {"LR", "DT", "RF", "SVM"}
    # Iris is easy; a real fit must clear a floor rather than guessing (0.33).
    assert out["holdout"]["RF"] > 0.85
    assert min(out["cv"].values()) > 0.85
    assert all(0.0 <= v <= 1.0 for v in out["holdout"].values())


def test_breast_cancer_experiment_produces_metrics():
    out = breast_cancer_experiment()
    assert out["n_samples"] == 569
    assert out["accuracy"] > 0.90
    assert out["roc_auc"] > 0.95
    assert "malignant" in out["report"] and "benign" in out["report"]
