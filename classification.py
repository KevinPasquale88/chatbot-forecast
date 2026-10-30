import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from function import EMBEDDING_MODEL, get_output_dirs


def _positive_class_index(classifier):
    return list(classifier.classes_).index(1)


def _optimal_f1_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-6)
    return thresholds[f1_scores.argmax()]


def training_classifier(dataset="heart_disease"):
    dirs = get_output_dirs(dataset)
    model_name = EMBEDDING_MODEL["model_name"]

    print(f"Evaluating model: {model_name}")
    X = np.load(os.path.join(dirs["embeddings"], EMBEDDING_MODEL["filename"]))
    y = np.load(os.path.join(dirs["embeddings"], EMBEDDING_MODEL["filename_label"]))

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = []

    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        inner_classifier = LogisticRegression(max_iter=2000)
        inner_scores = cross_val_predict(
            inner_classifier, X_train, y_train, cv=inner_cv, method="predict_proba"
        )[:, 1]
        tau = _optimal_f1_threshold(y_train, inner_scores)

        logistic_regression = LogisticRegression(max_iter=2000)
        logistic_regression.fit(X_train, y_train)
        y_score = logistic_regression.predict_proba(X_val)[:, _positive_class_index(logistic_regression)]
        y_pred = (y_score >= tau).astype(int)

        fold_metrics.append({
            "acc": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred, average='macro'),
            "auc": roc_auc_score(y_val, y_score),
            "tau": tau,
        })

    acc_mean = np.mean([m["acc"] for m in fold_metrics])
    f1_mean = np.mean([m["f1"] for m in fold_metrics])
    auc_mean = np.mean([m["auc"] for m in fold_metrics])
    tau_mean = np.mean([m["tau"] for m in fold_metrics])

    print(f"Accuracy (mean): {acc_mean:.4f}")
    print(f"Macro-F1 (mean): {f1_mean:.4f}")
    print(f"ROC-AUC (mean): {auc_mean:.4f}")
    print(f"Mean optimized threshold τ: {tau_mean:.4f}")

    df = pd.DataFrame({model_name: {"acc": acc_mean, "f1": f1_mean, "auc": auc_mean, "tau": tau_mean}}).T
    df.to_csv(os.path.join(dirs["results"], "model_performance.csv"))
    print(f"Saved to {os.path.join(dirs['results'], 'model_performance.csv')}")
