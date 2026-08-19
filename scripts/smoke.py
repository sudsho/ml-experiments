"""Offline smoke for ml-experiments.

Runs two representative experiments from this scratch pile end to end on CPU
with no downloads and no network. Both use sklearn built-in datasets, so there
is nothing to fetch and nothing to place in data/.

Experiment A: iris-classifier-comparison
    load_iris, compare LogisticRegression / DecisionTree / RandomForest / SVC
    on a held out split plus 5 fold cross validation.

Experiment B: breast-cancer-diagnosis
    load_breast_cancer, a StandardScaler + LogisticRegression pipeline, held
    out accuracy, ROC AUC, and a classification report. xgboost is used only if
    it is already importable (guarded); the smoke never installs it.

The core of each experiment is a plain function returning its metrics so the
pytest suite can reuse them. Running this file prints the metrics and asserts
they cleared sane floors, then prints SMOKE OK.
"""
from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, classification_report, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def iris_experiment():
    """Compare four classical models on the iris dataset.

    Returns a dict with per model held out accuracy and 5 fold cross
    validation means. Mirrors iris-classifier-comparison.ipynb.
    """
    data = load_iris()
    X, y = data.data, data.target
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "LR": LogisticRegression(max_iter=300),
        "DT": DecisionTreeClassifier(random_state=42),
        "RF": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(gamma="auto"),
    }

    holdout = {}
    cv = {}
    for name, model in models.items():
        model.fit(Xtr, ytr)
        holdout[name] = float(model.score(Xte, yte))
        cv[name] = float(cross_val_score(model, X, y, cv=5).mean())

    return {"holdout": holdout, "cv": cv, "n_samples": int(X.shape[0])}


def breast_cancer_experiment():
    """Scaled logistic regression on the breast cancer dataset.

    Returns held out accuracy, ROC AUC, and a text classification report.
    Mirrors breast-cancer-diagnosis.ipynb (core, xgboost guarded).
    """
    data = load_breast_cancer()
    X, y = data.data, data.target  # 0 = malignant, 1 = benign
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=0
    )

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000)),
        ]
    )
    pipe.fit(Xtr, ytr)

    accuracy = float(pipe.score(Xte, yte))
    probs = pipe.predict_proba(Xte)[:, 1]
    fpr, tpr, _ = roc_curve(yte, probs)
    roc_auc = float(auc(fpr, tpr))
    report = classification_report(
        yte, pipe.predict(Xte), target_names=["malignant", "benign"]
    )

    # Optional: an extra model comparison, only if xgboost is already present.
    # The smoke never installs it; a missing xgboost is fine.
    xgb_cv = None
    try:
        import xgboost as xgb

        clf = xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=0,
            eval_metric="logloss",
        )
        xgb_cv = float(cross_val_score(clf, X, y, cv=5).mean())
    except Exception:
        xgb_cv = None

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "report": report,
        "xgb_cv": xgb_cv,
        "n_samples": int(X.shape[0]),
    }


def main():
    np.random.seed(0)
    print("=" * 60)
    print("ml-experiments offline smoke")
    print("=" * 60)

    print("\n[A] iris-classifier-comparison (sklearn load_iris)")
    iris = iris_experiment()
    print(f"    samples: {iris['n_samples']}")
    print("    held out accuracy:")
    for name, score in iris["holdout"].items():
        print(f"      {name:4s} {score:.4f}")
    print("    5 fold cross validation mean:")
    for name, score in iris["cv"].items():
        print(f"      {name:4s} {score:.4f}")

    print("\n[B] breast-cancer-diagnosis (sklearn load_breast_cancer)")
    bc = breast_cancer_experiment()
    print(f"    samples: {bc['n_samples']}")
    print(f"    scaled LR held out accuracy: {bc['accuracy']:.4f}")
    print(f"    scaled LR ROC AUC:           {bc['roc_auc']:.4f}")
    if bc["xgb_cv"] is not None:
        print(f"    xgboost 5 fold CV mean:      {bc['xgb_cv']:.4f}")
    else:
        print("    xgboost not importable, skipped its comparison (expected offline)")
    print("    classification report:")
    for line in bc["report"].splitlines():
        print("      " + line)

    # Assertions: these are easy datasets, so real training must clear a floor.
    assert iris["holdout"]["RF"] > 0.85, iris["holdout"]
    assert min(iris["cv"].values()) > 0.85, iris["cv"]
    assert bc["accuracy"] > 0.90, bc["accuracy"]
    assert bc["roc_auc"] > 0.95, bc["roc_auc"]

    print("\n" + "=" * 60)
    print("SMOKE OK - both experiments trained and cleared metric floors")
    print("=" * 60)


if __name__ == "__main__":
    main()
