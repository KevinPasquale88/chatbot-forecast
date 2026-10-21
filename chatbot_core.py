import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from embedding import (
    CP_LABELS, RESTECG_LABELS, SLOPE_LABELS, THAL_LABELS,
    record_to_text_diabetes130, record_to_text_heart_disease,
)
from function import EMBEDDING_MODEL, get_output_dirs

DATASETS = ["heart_disease", "diabetes130"]

DATASET_LABELS = {
    "heart_disease": "❤️ Heart Disease",
    "diabetes130": "🏥 Diabetes Readmission",
}

WELCOME = (
    "Ciao! Sono un assistente che stima il rischio clinico in base ad alcuni dati di base.\n"
    "⚠️ Non sono uno strumento diagnostico: per qualsiasi dubbio consulta un medico.\n\n"
)

RESTART_WORDS = {"nuovo", "restart", "/start", "ricomincia"}


def _parse_choice(text, options):
    t = text.strip().lower()
    return options.get(t)


def _parse_int_range(text, low, high):
    try:
        value = int(float(text.strip().replace(",", ".")))
    except ValueError:
        return None
    return value if low <= value <= high else None


def _parse_float_range(text, low, high):
    try:
        value = float(text.strip().replace(",", "."))
    except ValueError:
        return None
    return value if low <= value <= high else None


def _parse_labeled(text, labels):
    t = text.strip()
    try:
        code = int(t)
    except ValueError:
        return None
    return code if code in labels else None


def _parse_nonempty(text):
    t = text.strip()
    return t if t else None


def _options_prompt(labels):
    return "\n".join(f"{code}) {label}" for code, label in labels.items())


QUESTIONS_HEART_DISEASE = [
    {"field": "sex", "prompt": "Sei uomo o donna? (rispondi 'uomo' o 'donna')",
     "parser": lambda t: _parse_choice(t, {"uomo": 1, "maschio": 1, "m": 1, "male": 1,
                                            "donna": 0, "femmina": 0, "f": 0, "female": 0}),
     "error": "Non ho capito. Rispondi 'uomo' o 'donna'."},
    {"field": "age", "prompt": "Quanti anni hai?",
     "parser": lambda t: _parse_int_range(t, 1, 120),
     "error": "Inserisci un'età valida (un numero tra 1 e 120)."},
    {"field": "cp", "prompt": "Che tipo di dolore al petto avverti?\n\n" + _options_prompt(CP_LABELS) +
                              "\n\nRispondi con il numero corrispondente.",
     "parser": lambda t: _parse_labeled(t, CP_LABELS),
     "error": "Rispondi con uno dei numeri elencati."},
    {"field": "trestbps", "prompt": "Qual è la tua pressione arteriosa a riposo (mmHg)?",
     "parser": lambda t: _parse_float_range(t, 60, 260),
     "error": "Inserisci un valore numerico di pressione tra 60 e 260 mmHg."},
    {"field": "chol", "prompt": "Qual è il tuo colesterolo sierico (mg/dl)?",
     "parser": lambda t: _parse_float_range(t, 50, 700),
     "error": "Inserisci un valore numerico di colesterolo tra 50 e 700 mg/dl."},
    {"field": "fbs", "prompt": "La tua glicemia a digiuno è superiore a 120 mg/dl? (si/no)",
     "parser": lambda t: _parse_choice(t, {"si": 1, "sì": 1, "yes": 1, "y": 1,
                                            "no": 0, "n": 0}),
     "error": "Rispondi 'si' o 'no'."},
    {"field": "restecg", "prompt": "Risultato dell'elettrocardiogramma a riposo?\n\n" + _options_prompt(RESTECG_LABELS) +
                                   "\n\nRispondi con il numero corrispondente.",
     "parser": lambda t: _parse_labeled(t, RESTECG_LABELS),
     "error": "Rispondi con uno dei numeri elencati."},
    {"field": "thalach", "prompt": "Qual è la frequenza cardiaca massima raggiunta?",
     "parser": lambda t: _parse_float_range(t, 60, 250),
     "error": "Inserisci un valore numerico tra 60 e 250."},
    {"field": "exang", "prompt": "Hai angina indotta dall'esercizio fisico? (si/no)",
     "parser": lambda t: _parse_choice(t, {"si": 1, "sì": 1, "yes": 1, "y": 1,
                                            "no": 0, "n": 0}),
     "error": "Rispondi 'si' o 'no'."},
    {"field": "oldpeak", "prompt": "Depressione del tratto ST indotta dall'esercizio (es. 1.5)?",
     "parser": lambda t: _parse_float_range(t, 0, 10),
     "error": "Inserisci un valore numerico tra 0 e 10."},
    {"field": "slope", "prompt": "Pendenza del tratto ST da sforzo massimo?\n\n" + _options_prompt(SLOPE_LABELS) +
                                 "\n\nRispondi con il numero corrispondente.",
     "parser": lambda t: _parse_labeled(t, SLOPE_LABELS),
     "error": "Rispondi con uno dei numeri elencati."},
    {"field": "ca", "prompt": "Numero di vasi principali colorati da fluoroscopia (0-3)?",
     "parser": lambda t: _parse_int_range(t, 0, 3),
     "error": "Inserisci un numero tra 0 e 3."},
    {"field": "thal", "prompt": "Talassemia?\n\n" + _options_prompt(THAL_LABELS) +
                                "\n\nRispondi con il numero corrispondente.",
     "parser": lambda t: _parse_labeled(t, THAL_LABELS),
     "error": "Rispondi con uno dei numeri elencati."},
]

QUESTIONS_DIABETES130 = [
    {"field": "race", "prompt": "Etnia del paziente (es. Caucasian, AfricanAmerican, Asian, Hispanic, Other)?",
     "parser": _parse_nonempty, "error": "Inserisci un valore non vuoto."},
    {"field": "gender", "prompt": "Genere del paziente? (rispondi 'uomo' o 'donna')",
     "parser": lambda t: _parse_choice(t, {"uomo": "Male", "maschio": "Male", "m": "Male", "male": "Male",
                                            "donna": "Female", "femmina": "Female", "f": "Female", "female": "Female"}),
     "error": "Rispondi 'uomo' o 'donna'."},
    {"field": "age", "prompt": "Fascia d'età (es. [50-60), [70-80))?",
     "parser": _parse_nonempty, "error": "Inserisci una fascia d'età valida."},
    {"field": "admission_type_id", "prompt": "ID tipo di ammissione (numero, es. 1)?",
     "parser": lambda t: _parse_int_range(t, 1, 30),
     "error": "Inserisci un numero valido."},
    {"field": "discharge_disposition_id", "prompt": "ID disposizione alla dimissione (numero, es. 1)?",
     "parser": lambda t: _parse_int_range(t, 1, 30),
     "error": "Inserisci un numero valido."},
    {"field": "admission_source_id", "prompt": "ID fonte di ammissione (numero, es. 1)?",
     "parser": lambda t: _parse_int_range(t, 1, 30),
     "error": "Inserisci un numero valido."},
    {"field": "time_in_hospital", "prompt": "Giorni di degenza in ospedale?",
     "parser": lambda t: _parse_int_range(t, 1, 60),
     "error": "Inserisci un numero valido di giorni."},
    {"field": "num_lab_procedures", "prompt": "Numero di esami di laboratorio effettuati?",
     "parser": lambda t: _parse_int_range(t, 0, 200),
     "error": "Inserisci un numero valido."},
    {"field": "num_procedures", "prompt": "Numero di procedure effettuate?",
     "parser": lambda t: _parse_int_range(t, 0, 50),
     "error": "Inserisci un numero valido."},
    {"field": "num_medications", "prompt": "Numero di farmaci somministrati?",
     "parser": lambda t: _parse_int_range(t, 0, 100),
     "error": "Inserisci un numero valido."},
    {"field": "number_outpatient", "prompt": "Visite ambulatoriali nell'anno precedente?",
     "parser": lambda t: _parse_int_range(t, 0, 100),
     "error": "Inserisci un numero valido."},
    {"field": "number_emergency", "prompt": "Accessi in pronto soccorso nell'anno precedente?",
     "parser": lambda t: _parse_int_range(t, 0, 100),
     "error": "Inserisci un numero valido."},
    {"field": "number_inpatient", "prompt": "Ricoveri nell'anno precedente?",
     "parser": lambda t: _parse_int_range(t, 0, 100),
     "error": "Inserisci un numero valido."},
    {"field": "number_diagnoses", "prompt": "Numero di diagnosi registrate?",
     "parser": lambda t: _parse_int_range(t, 1, 20),
     "error": "Inserisci un numero valido."},
    {"field": "max_glu_serum", "prompt": "Risultato test glicemia massima? (None, Norm, >200, >300)",
     "parser": _parse_nonempty, "error": "Inserisci un valore non vuoto."},
    {"field": "A1Cresult", "prompt": "Risultato test A1C? (None, Norm, >7, >8)",
     "parser": _parse_nonempty, "error": "Inserisci un valore non vuoto."},
    {"field": "insulin", "prompt": "Terapia insulinica? (No, Up, Down, Steady)",
     "parser": _parse_nonempty, "error": "Inserisci un valore non vuoto."},
    {"field": "change", "prompt": "Cambio di terapia durante il ricovero? (Ch/No)",
     "parser": _parse_nonempty, "error": "Inserisci un valore non vuoto."},
    {"field": "diabetesMed", "prompt": "Farmaco per il diabete prescritto? (Yes/No)",
     "parser": _parse_nonempty, "error": "Inserisci un valore non vuoto."},
]

QUESTIONS_BY_DATASET = {
    "heart_disease": QUESTIONS_HEART_DISEASE,
    "diabetes130": QUESTIONS_DIABETES130,
}

RECORD_TO_TEXT_BY_DATASET = {
    "heart_disease": record_to_text_heart_disease,
    "diabetes130": record_to_text_diabetes130,
}

RISK_LABEL_BY_DATASET = {
    "heart_disease": ("a RISCHIO di malattia cardiaca", "a BASSO rischio di malattia cardiaca"),
    "diabetes130": ("a RISCHIO di riammissione ospedaliera entro 30 giorni", "a BASSO rischio di riammissione ospedaliera entro 30 giorni"),
}

_classifier_cache = {}
_embedder = None


def _load_deployed_bundle(dataset):
    if dataset in _classifier_cache:
        return _classifier_cache[dataset]

    dirs = get_output_dirs(dataset)
    perf_path = os.path.join(dirs["results"], "model_performance.csv")
    embeddings_path = os.path.join(dirs["embeddings"], EMBEDDING_MODEL["filename"])
    labels_path = os.path.join(dirs["embeddings"], EMBEDDING_MODEL["filename_label"])
    if not (os.path.exists(perf_path) and os.path.exists(embeddings_path)):
        raise FileNotFoundError(
            f"Nessun risultato addestrato per '{dataset}'. "
            f"Esegui prima 'python main.py --dataset {dataset}'."
        )

    perf = pd.read_csv(perf_path, index_col=0)
    threshold = float(perf.loc[EMBEDDING_MODEL["model_name"], "tau"])

    X = np.load(embeddings_path)
    y = np.load(labels_path)

    classifier = LogisticRegression(max_iter=2000)
    classifier.fit(X, y)

    bundle = {"classifier": classifier, "threshold": threshold}
    _classifier_cache[dataset] = bundle
    return bundle


def _ensure_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL["name"])
    return _embedder


def _embed_text(text):
    return _ensure_embedder().encode([text], convert_to_numpy=True)[0]


def preload(dataset):
    _load_deployed_bundle(dataset)
    _ensure_embedder()


class ConversationSession:
    def __init__(self, dataset):
        self.dataset = dataset
        self.answers = {}
        self.step = 0

    @property
    def questions(self):
        return QUESTIONS_BY_DATASET[self.dataset]

    @property
    def is_complete(self):
        return self.step >= len(self.questions)


def dataset_selection_prompt():
    lines = ["Su quale dataset vuoi basare la valutazione?"]
    for key in DATASETS:
        lines.append(f"- {key}: {DATASET_LABELS[key]}")
    lines.append("Rispondi con il nome del dataset (es. 'heart_disease').")
    return "\n".join(lines)


def start_session(dataset):
    if dataset not in QUESTIONS_BY_DATASET:
        raise ValueError(f"Dataset sconosciuto: {dataset}")
    preload(dataset)
    session = ConversationSession(dataset)
    return session, WELCOME + session.questions[0]["prompt"]


def handle_message(session, user_text):
    if user_text.strip().lower() in RESTART_WORDS:
        session = ConversationSession(session.dataset)
        return session.questions[0]["prompt"], session

    if session.is_complete:
        session = ConversationSession(session.dataset)
        return "Iniziamo una nuova valutazione.\n\n" + session.questions[0]["prompt"], session

    question = session.questions[session.step]
    value = question["parser"](user_text)
    if value is None:
        return question["error"], session

    session.answers[question["field"]] = value
    session.step += 1

    if not session.is_complete:
        return session.questions[session.step]["prompt"], session

    return _predict(session), session


def _predict(session):
    bundle = _load_deployed_bundle(session.dataset)
    classifier = bundle["classifier"]
    threshold = bundle["threshold"]

    record_to_text = RECORD_TO_TEXT_BY_DATASET[session.dataset]
    text = record_to_text(session.answers)
    embedding = _embed_text(text)
    probability = classifier.predict_proba([embedding])[0, 1]
    risk = probability >= threshold

    risk_label, safe_label = RISK_LABEL_BY_DATASET[session.dataset]
    label = risk_label if risk else safe_label
    return (
        f"In base ai dati forniti, il paziente risulta {label}.\n"
        f"Probabilità stimata: {probability:.1%} (soglia di decisione: {threshold:.1%}).\n\n"
        "⚠️ Questo è solo un supporto informativo basato su un modello statistico, non una diagnosi medica. "
        "Consulta sempre un professionista sanitario.\n\n"
        "Scrivi 'nuovo' per fare un'altra valutazione."
    )
