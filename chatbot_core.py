import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from embedding import (
    CP_LABELS, RESTECG_LABELS, SLOPE_LABELS, THAL_LABELS,
    record_to_text_diabetes130, record_to_text_heart_disease,
)
from function import (
    EMBEDDING_MODEL, cat_cols, cat_cols_diabetes130, get_output_dirs,
    num_cols, num_cols_diabetes130,
)

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
UNKNOWN_ANSWERS = {"non lo so", "non so", "non ricordo", "sconosciuto", "ignoto", "n/a", "na", "?"}


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


def _question_prompt(question):
    return (
        f"{question['prompt']}\n\n"
        f"{question['explanation']}\n\n"
        "Se non lo sai, rispondi 'Non lo so'."
    )


def _is_unknown_answer(text):
    normalized = text.strip().lower().rstrip(".!?").strip()
    return normalized in UNKNOWN_ANSWERS


def _impute_unknown_answers(dataset, answers):
    if not any(pd.isna(value) for value in answers.values()):
        return answers

    dirs = get_output_dirs(dataset)
    training_data_path = os.path.join(dirs["preprocessing"], "X_train_raw.csv")
    if not os.path.exists(training_data_path):
        raise FileNotFoundError(
            f"Dati di training non trovati in '{training_data_path}', necessari per "
            "gestire le risposte sconosciute."
        )

    training_data = pd.read_csv(training_data_path)
    imputed_answers = answers.copy()
    numeric_columns = num_cols if dataset == "heart_disease" else num_cols_diabetes130
    categorical_columns = cat_cols if dataset == "heart_disease" else cat_cols_diabetes130

    for column in numeric_columns:
        if column in imputed_answers and pd.isna(imputed_answers[column]):
            imputed_answers[column] = training_data[column].median()
    for column in categorical_columns:
        if column in imputed_answers and pd.isna(imputed_answers[column]):
            imputed_answers[column] = training_data[column].mode().iloc[0]
    return imputed_answers


def _positive_probability(classifier, embedding):
    positive_index = list(classifier.classes_).index(1)
    return classifier.predict_proba([embedding])[0, positive_index]


QUESTIONS_HEART_DISEASE = [
    {"field": "sex", "prompt": "Per la persona valutata, quale sesso è indicato nei dati: uomo o donna?",
     "parser": lambda t: _parse_choice(t, {"uomo": 1, "maschio": 1, "m": 1, "male": 1,
                                            "donna": 0, "femmina": 0, "f": 0, "female": 0}),
     "error": "Non ho capito. Rispondi 'uomo' o 'donna'."},
    {"field": "age", "prompt": "Quanti anni ha la persona?",
     "parser": lambda t: _parse_int_range(t, 1, 120),
     "error": "Inserisci un'età valida (un numero tra 1 e 120)."},
    {"field": "cp", "prompt": "Quale descrizione del dolore o fastidio al petto è riportata nel referto?\n\n" + _options_prompt(CP_LABELS) +
                              "\n\nRispondi con il numero corrispondente.",
     "parser": lambda t: _parse_labeled(t, CP_LABELS),
     "error": "Rispondi con uno dei numeri elencati."},
    {"field": "trestbps", "prompt": "Qual è il valore della pressione misurata a riposo (mmHg)?",
     "parser": lambda t: _parse_float_range(t, 60, 260),
     "error": "Inserisci un valore numerico di pressione tra 60 e 260 mmHg."},
    {"field": "chol", "prompt": "Qual è il valore del colesterolo riportato nell'esame del sangue (mg/dl)?",
     "parser": lambda t: _parse_float_range(t, 50, 700),
     "error": "Inserisci un valore numerico di colesterolo tra 50 e 700 mg/dl."},
    {"field": "fbs", "prompt": "Nel test fatto a digiuno, la glicemia supera 120 mg/dl? (sì/no)",
     "parser": lambda t: _parse_choice(t, {"si": 1, "sì": 1, "yes": 1, "y": 1,
                                            "no": 0, "n": 0}),
     "error": "Rispondi 'si' o 'no'."},
    {"field": "restecg", "prompt": "Quale risultato è riportato nell'elettrocardiogramma (ECG) a riposo?\n\n" + _options_prompt(RESTECG_LABELS) +
                                   "\n\nRispondi con il numero corrispondente.",
     "parser": lambda t: _parse_labeled(t, RESTECG_LABELS),
     "error": "Rispondi con uno dei numeri elencati."},
    {"field": "thalach", "prompt": "Durante la prova da sforzo, qual è stato il numero massimo di battiti al minuto?",
     "parser": lambda t: _parse_float_range(t, 60, 250),
     "error": "Inserisci un valore numerico tra 60 e 250."},
    {"field": "exang", "prompt": "Durante la prova da sforzo, è comparso dolore o oppressione al petto? (sì/no)",
     "parser": lambda t: _parse_choice(t, {"si": 1, "sì": 1, "yes": 1, "y": 1,
                                            "no": 0, "n": 0}),
     "error": "Rispondi 'si' o 'no'."},
    {"field": "oldpeak", "prompt": "Nel referto della prova da sforzo, quale valore è riportato per il tratto ST? (es. 1,5)",
     "parser": lambda t: _parse_float_range(t, 0, 10),
     "error": "Inserisci un valore numerico tra 0 e 10."},
    {"field": "slope", "prompt": "Nel referto, quale andamento è indicato per il tratto ST durante lo sforzo?\n\n" + _options_prompt(SLOPE_LABELS) +
                                 "\n\nRispondi con il numero corrispondente.",
     "parser": lambda t: _parse_labeled(t, SLOPE_LABELS),
     "error": "Rispondi con uno dei numeri elencati."},
    {"field": "ca", "prompt": "Nel referto dell'esame con contrasto, quanti vasi principali sono indicati? (0-3)",
     "parser": lambda t: _parse_int_range(t, 0, 3),
     "error": "Inserisci un numero tra 0 e 3."},
    {"field": "thal", "prompt": "Quale risultato compare alla voce 'thal' nel referto cardiologico?\n\n" + _options_prompt(THAL_LABELS) +
                                "\n\nRispondi con il numero corrispondente.",
     "parser": lambda t: _parse_labeled(t, THAL_LABELS),
     "error": "Rispondi con uno dei numeri elencati."},
]

QUESTIONS_DIABETES130 = [
    {"field": "race", "prompt": "Quale gruppo etnico è riportato nella cartella clinica? (es. Caucasian, AfricanAmerican, Asian, Hispanic, Other)",
     "parser": _parse_nonempty, "error": "Inserisci un valore non vuoto."},
    {"field": "gender", "prompt": "Nei dati della persona, il genere è indicato come uomo o donna?",
     "parser": lambda t: _parse_choice(t, {"uomo": "Male", "maschio": "Male", "m": "Male", "male": "Male",
                                            "donna": "Female", "femmina": "Female", "f": "Female", "female": "Female"}),
     "error": "Rispondi 'uomo' o 'donna'."},
    {"field": "age", "prompt": "Quale fascia d'età è indicata nella cartella? (es. [50-60))",
     "parser": _parse_nonempty, "error": "Inserisci una fascia d'età valida."},
    {"field": "admission_type_id", "prompt": "Quale codice è riportato per il tipo di ricovero? (es. 1)",
     "parser": lambda t: _parse_int_range(t, 1, 30),
     "error": "Inserisci un numero valido."},
    {"field": "discharge_disposition_id", "prompt": "Quale codice descrive dove è stata mandata la persona alla dimissione? (es. 1)",
     "parser": lambda t: _parse_int_range(t, 1, 30),
     "error": "Inserisci un numero valido."},
    {"field": "admission_source_id", "prompt": "Quale codice indica da dove è arrivata la persona in ospedale? (es. 1)",
     "parser": lambda t: _parse_int_range(t, 1, 30),
     "error": "Inserisci un numero valido."},
    {"field": "time_in_hospital", "prompt": "Per quanti giorni la persona è rimasta in ospedale?",
     "parser": lambda t: _parse_int_range(t, 1, 60),
     "error": "Inserisci un numero valido di giorni."},
    {"field": "num_lab_procedures", "prompt": "Quanti esami di laboratorio sono stati fatti durante il ricovero?",
     "parser": lambda t: _parse_int_range(t, 0, 200),
     "error": "Inserisci un numero valido."},
    {"field": "num_procedures", "prompt": "Quante procedure o interventi sono stati effettuati durante il ricovero?",
     "parser": lambda t: _parse_int_range(t, 0, 50),
     "error": "Inserisci un numero valido."},
    {"field": "num_medications", "prompt": "Quanti farmaci sono stati somministrati o prescritti durante il ricovero?",
     "parser": lambda t: _parse_int_range(t, 0, 100),
     "error": "Inserisci un numero valido."},
    {"field": "number_outpatient", "prompt": "Quante visite ambulatoriali ha fatto la persona nell'anno prima del ricovero?",
     "parser": lambda t: _parse_int_range(t, 0, 100),
     "error": "Inserisci un numero valido."},
    {"field": "number_emergency", "prompt": "Quante volte la persona è andata al pronto soccorso nell'anno prima del ricovero?",
     "parser": lambda t: _parse_int_range(t, 0, 100),
     "error": "Inserisci un numero valido."},
    {"field": "number_inpatient", "prompt": "Quanti altri ricoveri ha avuto la persona nell'anno precedente?",
     "parser": lambda t: _parse_int_range(t, 0, 100),
     "error": "Inserisci un numero valido."},
    {"field": "number_diagnoses", "prompt": "Quante diagnosi sono elencate nella cartella del ricovero?",
     "parser": lambda t: _parse_int_range(t, 1, 20),
     "error": "Inserisci un numero valido."},
    {"field": "max_glu_serum", "prompt": "Quale risultato è riportato per il test della glicemia? (None, Norm, >200, >300)",
     "parser": _parse_nonempty, "error": "Inserisci un valore non vuoto."},
    {"field": "A1Cresult", "prompt": "Quale risultato è riportato per il test A1C? (None, Norm, >7, >8)",
     "parser": _parse_nonempty, "error": "Inserisci un valore non vuoto."},
    {"field": "insulin", "prompt": "Come è indicata la terapia con insulina nella cartella? (No, Up, Down, Steady)",
     "parser": _parse_nonempty, "error": "Inserisci un valore non vuoto."},
    {"field": "change", "prompt": "Durante il ricovero è stata cambiata la terapia per il diabete? (Ch/No)",
     "parser": _parse_nonempty, "error": "Inserisci un valore non vuoto."},
    {"field": "diabetesMed", "prompt": "È stato prescritto un farmaco per il diabete? (Yes/No)",
     "parser": _parse_nonempty, "error": "Inserisci un valore non vuoto."},
]

QUESTION_EXPLANATIONS = {
    "heart_disease": {
        "sex": "Usa il valore riportato nei dati clinici della persona.",
        "age": "Inserisci gli anni compiuti, non l'anno di nascita.",
        "cp": "Nel referto, 'angina' indica un dolore o una pressione al petto. Scegli la descrizione più vicina a quella riportata; 'asintomatico' significa che non sono stati registrati sintomi.",
        "trestbps": "È la pressione misurata mentre la persona è tranquilla, senza aver appena fatto attività fisica. Di solito il valore si trova in un referto.",
        "chol": "È il risultato del colesterolo misurato con un prelievo del sangue. Riporta il numero indicato nell'esame.",
        "fbs": "Il test è 'a digiuno' se è stato fatto dopo diverse ore senza mangiare. Se il referto riporta 120 o meno, rispondi 'no'.",
        "restecg": "L'ECG registra l'attività elettrica del cuore. Scegli il risultato scritto nel referto; 'ipertrofia' indica un ispessimento di una parte del cuore.",
        "thalach": "È il valore più alto registrato durante una prova da sforzo, espresso in battiti al minuto. Puoi trovarlo nel referto.",
        "exang": "Considera il dolore comparso durante la prova da sforzo, non quello presente a riposo.",
        "oldpeak": "È un numero misurato nel tracciato del test da sforzo. Se hai il referto, copialo così com'è, usando il punto o la virgola decimale.",
        "slope": "Descrive se una linea del tracciato sale, resta piatta o scende. Scegli il termine riportato nel referto, senza dover interpretare l'esame.",
        "ca": "È il numero di vasi sanguigni principali messi in evidenza durante un esame con contrasto. Il dato è riportato nel referto dell'esame.",
        "thal": "È una voce tecnica del referto cardiologico. Scegli il risultato indicato nel documento; non occorre dedurlo dai sintomi.",
    },
    "diabetes130": {
        "race": "Riporta la categoria così come compare nella cartella. Se non è indicata, puoi rispondere 'Non lo so'.",
        "gender": "Usa la categoria riportata nei dati della persona.",
        "age": "La fascia [50-60) significa da 50 a 59 anni; la fascia successiva inizia a 60 anni.",
        "admission_type_id": "È un codice interno dell'ospedale: per esempio, 1 significa ricovero d'emergenza, 2 urgente e 3 programmato. Se puoi, controlla il documento del ricovero.",
        "discharge_disposition_id": "È un codice che indica dove è stata mandata la persona dopo il ricovero, per esempio a casa o in un'altra struttura. Lo trovi nella documentazione di dimissione.",
        "admission_source_id": "È un codice che indica da dove è arrivata la persona, per esempio dal pronto soccorso o da un altro ospedale. Controlla la cartella del ricovero.",
        "time_in_hospital": "Conta i giorni trascorsi in ospedale durante questo ricovero.",
        "num_lab_procedures": "Conta gli esami di laboratorio, come le analisi del sangue, registrati durante il ricovero.",
        "num_procedures": "Conta le procedure o gli interventi registrati durante il ricovero; non include i normali esami di laboratorio.",
        "num_medications": "Inserisci il numero totale registrato nella cartella, non il nome dei farmaci.",
        "number_outpatient": "Conta le visite senza ricovero nei 12 mesi prima di questo episodio.",
        "number_emergency": "Conta gli accessi al pronto soccorso nei 12 mesi prima di questo episodio.",
        "number_inpatient": "Conta i ricoveri ospedalieri nei 12 mesi prima di questo episodio.",
        "number_diagnoses": "Conta le diagnosi elencate per questo ricovero, non solo quella principale.",
        "max_glu_serum": "'Norm' indica un risultato nella norma; '>200' e '>300' indicano valori oltre quelle soglie. 'None' significa che il test non è stato registrato.",
        "A1Cresult": "Il test A1C dà un'indicazione media della glicemia degli ultimi mesi. 'Norm' è nella norma; '>7' e '>8' indicano valori oltre queste soglie; 'None' significa che il test non è stato registrato.",
        "insulin": "'Up' significa dose aumentata, 'Down' ridotta, 'Steady' invariata e 'No' nessuna terapia con insulina registrata.",
        "change": "'Ch' significa che la terapia è stata cambiata; 'No' che non risulta cambiata.",
        "diabetesMed": "Rispondi in base alla cartella del ricovero: 'Yes' se è stato prescritto un farmaco per il diabete, 'No' altrimenti.",
    },
}

QUESTIONS_BY_DATASET = {
    "heart_disease": QUESTIONS_HEART_DISEASE,
    "diabetes130": QUESTIONS_DIABETES130,
}

for dataset, questions in QUESTIONS_BY_DATASET.items():
    for question in questions:
        question["explanation"] = QUESTION_EXPLANATIONS[dataset][question["field"]]

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
    return session, WELCOME + _question_prompt(session.questions[0])


def handle_message(session, user_text):
    if user_text.strip().lower() in RESTART_WORDS:
        session = ConversationSession(session.dataset)
        return _question_prompt(session.questions[0]), session

    if session.is_complete:
        session = ConversationSession(session.dataset)
        return "Iniziamo una nuova valutazione.\n\n" + _question_prompt(session.questions[0]), session

    question = session.questions[session.step]
    value = np.nan if _is_unknown_answer(user_text) else question["parser"](user_text)
    if value is None:
        return question["error"] + "\n\nSe non lo sai, rispondi 'Non lo so'.", session

    session.answers[question["field"]] = value
    session.step += 1

    if not session.is_complete:
        return _question_prompt(session.questions[session.step]), session

    return _predict(session), session


def _predict(session):
    bundle = _load_deployed_bundle(session.dataset)
    classifier = bundle["classifier"]
    threshold = bundle["threshold"]

    record_to_text = RECORD_TO_TEXT_BY_DATASET[session.dataset]
    answers = _impute_unknown_answers(session.dataset, session.answers)
    text = record_to_text(answers)
    embedding = _embed_text(text)
    probability = _positive_probability(classifier, embedding)
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
