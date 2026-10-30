import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from embedding import record_to_text_diabetes130, record_to_text_heart_disease
from function import EMBEDDING_MODEL, load_diabetes130, load_heart_disease

# Same protocol as the deployed model. The encoder is judged on the cross-validation score only:
# the test score is printed for the record and must not be used to pick a winner.
ENCODERS = [
    EMBEDDING_MODEL["name"],
    "NeuML/pubmedbert-base-embeddings",
    "emilyalsentzer/Bio_ClinicalBERT",
]
TASKS = [
    ("heart_disease", load_heart_disease, "num", record_to_text_heart_disease),
    ("diabetes130", load_diabetes130, "readmitted", record_to_text_diabetes130),
]


def main():
    for encoder_name in ENCODERS:
        model = SentenceTransformer(encoder_name)
        model.max_seq_length = EMBEDDING_MODEL["max_seq_length"]

        for dataset, load, target, to_text in TASKS:
            data = load()
            X = data.drop(target, axis=1)
            y = (data[target] > 0).astype(int).to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            train = model.encode([to_text(r) for _, r in X_train.iterrows()], batch_size=32)
            test = model.encode([to_text(r) for _, r in X_test.iterrows()], batch_size=32)

            folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv = cross_val_score(
                LogisticRegression(max_iter=2000), train, y_train, cv=folds, scoring="roc_auc"
            )
            classifier = LogisticRegression(max_iter=2000).fit(train, y_train)
            test_auc = roc_auc_score(y_test, classifier.predict_proba(test)[:, 1])
            print(f"{encoder_name} | {dataset} | CV {np.mean(cv):.3f} ± {np.std(cv):.3f} | test {test_auc:.3f}")


if __name__ == "__main__":
    main()
