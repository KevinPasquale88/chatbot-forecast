import numpy as np
import pandas as pd

import chatbot_core
import classification
from chatbot_core import ConversationSession, handle_message


def test_every_question_includes_an_explanation():
    for questions in chatbot_core.QUESTIONS_BY_DATASET.values():
        for question in questions:
            explanation = question["explanation"]
            assert explanation.strip()
            assert explanation in chatbot_core._question_prompt(question)
            assert "Non lo so" in chatbot_core._question_prompt(question)


def test_unknown_answer_advances_questionnaire():
    session = ConversationSession("heart_disease")

    reply, updated_session = handle_message(session, "Non lo so")

    assert updated_session.step == 1
    assert np.isnan(updated_session.answers["sex"])
    assert "Non lo so" in reply


def test_invalid_answer_reminds_user_of_unknown_option():
    session = ConversationSession("heart_disease")

    reply, updated_session = handle_message(session, "forse")

    assert updated_session.step == 0
    assert "Non lo so" in reply


def test_unknown_values_are_imputed_from_training_data(tmp_path, monkeypatch):
    training_dir = tmp_path / "preprocessing"
    training_dir.mkdir()
    pd.DataFrame({"age": [40, 60], "sex": [0, 1]}).to_csv(
        training_dir / "X_train_raw.csv", index=False
    )
    monkeypatch.setattr(
        chatbot_core,
        "get_output_dirs",
        lambda dataset: {"preprocessing": str(training_dir)},
    )

    result = chatbot_core._impute_unknown_answers(
        "heart_disease", {"age": np.nan, "sex": np.nan}
    )

    assert result == {"age": 50, "sex": 0}


def test_prediction_uses_probability_for_positive_class():
    class Classifier:
        classes_ = np.array([1, 0])

        def predict_proba(self, rows):
            return np.array([[0.8, 0.2]])

    assert chatbot_core._positive_probability(Classifier(), np.array([0.0])) == 0.8


def test_training_classifier_calibrates_threshold_separately(tmp_path, monkeypatch):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    rng = np.random.default_rng(42)
    features = np.vstack((rng.normal(-1, 0.5, (50, 4)), rng.normal(1, 0.5, (50, 4))))
    labels = np.array([0] * 50 + [1] * 50)

    monkeypatch.setattr(
        classification,
        "get_output_dirs",
        lambda dataset: {"embeddings": str(tmp_path), "results": str(results_dir)},
    )
    monkeypatch.setattr(
        classification.np,
        "load",
        lambda path: labels if path.endswith("_labels.npy") else features,
    )

    classification.training_classifier("heart_disease")

    metrics = pd.read_csv(results_dir / "model_performance.csv", index_col=0)
    assert 0 < metrics.loc["sentence-biobert", "tau"] < 1
    assert 0 <= metrics.loc["sentence-biobert", "f1"] <= 1