import numpy as np
import pandas as pd

import chatbot_core
import classification
import embedding
import function
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


def test_unknown_values_become_not_recorded_in_the_sentence():
    answers = {question["field"]: np.nan for question in chatbot_core.QUESTIONS_HEART_DISEASE}

    text = chatbot_core.record_to_text_heart_disease(answers)

    assert "serum cholesterol: not recorded" in text
    assert "nan" not in text


def test_too_many_unknown_answers_give_no_estimate():
    session = ConversationSession("heart_disease")

    for _ in session.questions:
        reply, session = handle_message(session, "Non lo so")

    assert "13 domande su 13" in reply
    assert "Probabilità" not in reply


def test_question_mark_counts_as_unknown():
    assert chatbot_core._is_unknown_answer("?")
    assert chatbot_core._is_unknown_answer("Non lo so.")


def test_int_parser_rejects_infinity_and_decimals():
    assert chatbot_core._parse_int_range("inf", 1, 120) is None
    assert chatbot_core._parse_int_range("1e400", 1, 120) is None
    assert chatbot_core._parse_int_range("63.9", 1, 120) is None
    assert chatbot_core._parse_int_range("63", 1, 120) == 63


def test_diabetes_categories_are_validated_and_normalized():
    parsers = {q["field"]: q["parser"] for q in chatbot_core.QUESTIONS_DIABETES130}

    assert parsers["race"]("caucasian") == "Caucasian"
    assert parsers["race"]("boh") is None
    assert parsers["A1Cresult"]("none") == "None"
    assert parsers["age"]("[50-60)") == "[50-60)"
    assert parsers["discharge_disposition_id"]("11") is None


def test_embedder_reads_the_whole_sentence(monkeypatch):
    class FakeEncoder:
        def __init__(self, name):
            self.max_seq_length = 100

    monkeypatch.setattr(embedding, "SentenceTransformer", FakeEncoder)

    assert embedding.load_embedder().max_seq_length == 512


def test_none_stays_a_category_in_diabetes():
    df = function.load_diabetes130(sample_size=2000)

    assert "None" in set(df["A1Cresult"])
    assert df["A1Cresult"].notna().all()


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
    assert 0 <= metrics.loc["sentence-biobert", "test_f1"] <= 1
    assert (results_dir / "deployed_bundle.joblib").exists()