import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from function import EMBEDDING_MODEL, get_output_dirs


def _positive_class_index(classifier):
    return list(classifier.classes_).index(1)


def _optimal_f1_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-6)
    return thresholds[f1_scores.argmax()]


def _out_of_fold_threshold(X, y):
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_predict(
        LogisticRegression(max_iter=2000), X, y, cv=inner_cv, method="predict_proba"
    )[:, 1]
    return _optimal_f1_threshold(y, scores)


def training_classifier(dataset="heart_disease"):
    dirs = get_output_dirs(dataset)
    model_name = EMBEDDING_MODEL["model_name"]

    print(f"Evaluating model: {model_name}")
    X = np.load(os.path.join(dirs["embeddings"], EMBEDDING_MODEL["filename"]))
    y = np.load(os.path.join(dirs["embeddings"], EMBEDDING_MODEL["filename_label"]))
    X_test = np.load(os.path.join(dirs["embeddings"], EMBEDDING_MODEL["filename_test"]))
    y_test = np.load(os.path.join(dirs["embeddings"], EMBEDDING_MODEL["filename_test_label"]))

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        tau = _out_of_fold_threshold(X_train, y_train)

        logistic_regression = LogisticRegression(max_iter=2000)
        logistic_regression.fit(X_train, y_train)
        y_score = logistic_regression.predict_proba(X_val)[:, _positive_class_index(logistic_regression)]
        y_pred = (y_score >= tau).astype(int)

        fold_metrics.append({
            "acc": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred, average='macro'),
            "auc": roc_auc_score(y_val, y_score),
        })

    # Final protocol: threshold from out-of-fold scores on the whole training set, classifier fitted
    # on the same data, then a single measurement on the untouched test set.
    threshold = _out_of_fold_threshold(X, y)
    classifier = LogisticRegression(max_iter=2000)
    classifier.fit(X, y)
    test_score = classifier.predict_proba(X_test)[:, _positive_class_index(classifier)]
    test_pred = (test_score >= threshold).astype(int)

    metrics = {
        "cv_acc": np.mean([m["acc"] for m in fold_metrics]),
        "cv_f1": np.mean([m["f1"] for m in fold_metrics]),
        "cv_auc": np.mean([m["auc"] for m in fold_metrics]),
        "test_acc": accuracy_score(y_test, test_pred),
        "test_f1": f1_score(y_test, test_pred, average='macro'),
        "test_auc": roc_auc_score(y_test, test_score),
        "test_pr_auc": average_precision_score(y_test, test_score),
        "test_precision": precision_score(y_test, test_pred, zero_division=0),
        "test_recall": recall_score(y_test, test_pred),
        "test_brier": brier_score_loss(y_test, test_score),
        "tau": threshold,
    }
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    df = pd.DataFrame({model_name: metrics}).T
    df.to_csv(os.path.join(dirs["results"], "model_performance.csv"))
    print(f"Saved to {os.path.join(dirs['results'], 'model_performance.csv')}")

    joblib.dump(
        {
            "classifier": classifier,
            "threshold": float(threshold),
            "encoder": EMBEDDING_MODEL["name"],
            "max_seq_length": EMBEDDING_MODEL["max_seq_length"],
        },
        os.path.join(dirs["results"], EMBEDDING_MODEL["bundle_filename"]),
    )
