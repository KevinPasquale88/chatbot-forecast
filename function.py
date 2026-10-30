import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]

# Diabetes 130-US Hospitals dataset (UCI). The raw file has ~50 columns; most are
# near-constant drug-dosage flags (e.g. examide, citoglipton) or high-missingness
# identifiers (weight, payer_code, patient_nbr) that add no signal, so only a
# clinically meaningful subset is kept here.
columns_diabetes130 = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses", "max_glu_serum", "A1Cresult", "insulin", "change",
    "diabetesMed", "readmitted"
]
# Discharge to hospice or death: the patient cannot be readmitted, so the outcome is not defined.
hospice_or_death_discharge_ids = [11, 13, 14, 19, 20, 21]

# All embeddings are generated with one biomedical sentence-transformer. confronto_encoder.py
# compares it with two alternatives: the three are equivalent within the margin of error.
EMBEDDING_MODEL = {
    "model_name": "sentence-biobert",
    "name": "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
    "filename": "sentence_biobert_embeddings.npy",
    "filename_label": "sentence_biobert_embeddings_labels.npy",
    "filename_test": "sentence_biobert_test_embeddings.npy",
    "filename_test_label": "sentence_biobert_test_embeddings_labels.npy",
    "bundle_filename": "deployed_bundle.joblib",
    # The model default is 100 tokens: the clinical sentences are longer and the last fields
    # would be silently dropped. 512 is BERT's positional limit.
    "max_seq_length": 512,
}

datasets = ["heart_disease", "diabetes130"]

# Per-dataset output tree: each dataset's own preprocessing/embeddings/results live under
# datas/<dataset>/, so the two benchmarks never overwrite each other.
def get_output_dirs(dataset):
    if dataset not in datasets:
        raise ValueError(f"Invalid dataset '{dataset}'. Valid options are: {datasets}")

    base = os.path.join("datas", dataset)
    dirs = {
        "preprocessing": os.path.join(base, "preprocessing"),
        "embeddings": os.path.join(base, "embeddings"),
        "results": os.path.join(base, "results"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

# Wipes this dataset's own preprocessing/embeddings/results folders, so each rebuild starts
# from a clean slate instead of mixing in files left over from a previous run (e.g. a stale
# model_performance.csv from before a code change).
def clean_output_dirs(dataset):
    dirs = get_output_dirs(dataset)
    for folder in dirs.values():
        for file_name in os.listdir(folder):
            full_path = os.path.join(folder, file_name)
            if os.path.isfile(full_path):
                os.remove(full_path)
                print(f"[Clean] Removed {full_path}")

#load data from files and concatenate into one dataframe
def load_heart_disease():
    files = [
        "datasets/heart+disease/processed.cleveland.data",
        "datasets/heart+disease/processed.hungarian.data",
        "datasets/heart+disease/processed.switzerland.data",
        "datasets/heart+disease/processed.va.data"
    ]
    dfs = [pd.read_csv(f, header=None, na_values="?") for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df.columns = columns

    # In some source files (notably Switzerland and Hungary) missing cholesterol/resting
    # blood pressure readings are encoded as 0 rather than "?". 0 mg/dl or 0 mm Hg is not a
    # physiologically valid value for either, so treat it as missing like the rest of the pipeline.
    df["chol"] = df["chol"].replace(0, np.nan)
    df["trestbps"] = df["trestbps"].replace(0, np.nan)

    return df

#load and sample the Diabetes 130-US Hospitals dataset from a local CSV
def load_diabetes130(sample_size=20000, random_state=42):
    # keep_default_na=False: "None" means "test not performed" here, it is not a missing value.
    df = pd.read_csv(
        "datasets/diabetes+130-us+hospitals+for+years+1999-2008/diabetic_data.csv",
        na_values="?", keep_default_na=False,
    )
    df = df[~df["discharge_disposition_id"].isin(hospice_or_death_discharge_ids)]
    # One encounter per patient (the first), so nobody ends up in both training and test.
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr")
    df = df[columns_diabetes130].copy()
    df["gender"] = df["gender"].replace("Unknown/Invalid", np.nan)

    # Binarize to the standard early-readmission benchmark task for this dataset:
    # 1 = readmitted within 30 days, 0 = readmitted later or not at all.
    df["readmitted"] = (df["readmitted"] == "<30").astype(int)

    if sample_size is not None and sample_size < len(df):
        df, _ = train_test_split(
            df, train_size=sample_size, stratify=df["readmitted"], random_state=random_state
        )
        df = df.reset_index(drop=True)

    return df
