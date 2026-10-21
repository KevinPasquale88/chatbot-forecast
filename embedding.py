import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from function import EMBEDDING_MODEL, get_output_dirs

# Standard UCI Heart Disease attribute encodings
CP_LABELS = {1: "angina tipica", 2: "angina atipica", 3: "dolore non anginoso", 4: "asintomatico"}
RESTECG_LABELS = {0: "normale", 1: "anomalia dell'onda ST-T", 2: "ipertrofia ventricolare sinistra"}
SLOPE_LABELS = {1: "ascendente", 2: "piatto", 3: "discendente"}
THAL_LABELS = {3: "normale", 6: "difetto fisso", 7: "difetto reversibile"}

def _fmt_num(value, unit="", ndigits=1):
    if pd.isna(value):
        return "not recorded"
    return f"{round(float(value), ndigits):g}{unit}"

def _fmt_cat(value, labels):
    if pd.isna(value):
        return "not recorded"
    return labels.get(int(round(float(value))), "unknown")

def _fmt_bool(value, true_label, false_label):
    if pd.isna(value):
        return "not recorded"
    return true_label if int(round(float(value))) == 1 else false_label

def record_to_text_heart_disease(row):
    sex = _fmt_bool(row["sex"], "Male", "Female")
    parts = [
        f"{sex} patient, {_fmt_num(row['age'], ndigits=0)} years old",
        f"chest pain type: {_fmt_cat(row['cp'], CP_LABELS)}",
        f"resting blood pressure: {_fmt_num(row['trestbps'], ' mm Hg', ndigits=0)}",
        f"serum cholesterol: {_fmt_num(row['chol'], ' mg/dl', ndigits=0)}",
        f"fasting blood sugar > 120 mg/dl: {_fmt_bool(row['fbs'], 'yes', 'no')}",
        f"resting electrocardiographic results: {_fmt_cat(row['restecg'], RESTECG_LABELS)}",
        f"maximum heart rate achieved: {_fmt_num(row['thalach'], ndigits=0)}",
        f"exercise-induced angina: {_fmt_bool(row['exang'], 'yes', 'no')}",
        f"ST depression induced by exercise: {_fmt_num(row['oldpeak'])}",
        f"slope of the peak exercise ST segment: {_fmt_cat(row['slope'], SLOPE_LABELS)}",
        f"number of major vessels colored by fluoroscopy: {_fmt_num(row['ca'], ndigits=0)}",
        f"thalassemia: {_fmt_cat(row['thal'], THAL_LABELS)}",
    ]
    return ", ".join(parts)

def _fmt_raw(value):
    return "not recorded" if pd.isna(value) else str(value)

def record_to_text_diabetes130(row):
    parts = [
        f"{_fmt_raw(row['gender'])} patient, age range {_fmt_raw(row['age'])}",
        f"race: {_fmt_raw(row['race'])}",
        f"admission type id: {_fmt_raw(row['admission_type_id'])}",
        f"discharge disposition id: {_fmt_raw(row['discharge_disposition_id'])}",
        f"admission source id: {_fmt_raw(row['admission_source_id'])}",
        f"time in hospital: {_fmt_raw(row['time_in_hospital'])} days",
        f"number of lab procedures: {_fmt_raw(row['num_lab_procedures'])}",
        f"number of procedures: {_fmt_raw(row['num_procedures'])}",
        f"number of medications: {_fmt_raw(row['num_medications'])}",
        f"outpatient visits in prior year: {_fmt_raw(row['number_outpatient'])}",
        f"emergency visits in prior year: {_fmt_raw(row['number_emergency'])}",
        f"inpatient visits in prior year: {_fmt_raw(row['number_inpatient'])}",
        f"number of diagnoses: {_fmt_raw(row['number_diagnoses'])}",
        f"max glucose serum test result: {_fmt_raw(row['max_glu_serum'])}",
        f"A1C test result: {_fmt_raw(row['A1Cresult'])}",
        f"insulin therapy: {_fmt_raw(row['insulin'])}",
        f"medication changed during encounter: {_fmt_raw(row['change'])}",
        f"prescribed diabetes medication: {_fmt_raw(row['diabetesMed'])}",
    ]
    return ", ".join(parts)

def embeddings(X, y, dataset="heart_disease"):
    dirs = get_output_dirs(dataset)
    record_to_text = record_to_text_diabetes130 if dataset == "diabetes130" else record_to_text_heart_disease
    texts = [record_to_text(r) for _, r in X.iterrows()]

    print(f"[Embedding] Generating {EMBEDDING_MODEL['model_name']} embeddings for {len(texts)} records...")
    model = SentenceTransformer(EMBEDDING_MODEL["name"])
    vectors = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    np.save(os.path.join(dirs["embeddings"], EMBEDDING_MODEL["filename"]), vectors.astype(np.float32))
    np.save(os.path.join(dirs["embeddings"], EMBEDDING_MODEL["filename_label"]), np.asarray(y, dtype=np.int32))
    print(f"[OK] Saved embeddings → {dirs['embeddings']}")
