import os

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC

from function import (
    load_heart_disease, load_diabetes130,
    num_cols, cat_cols,
    num_cols_diabetes130, cat_cols_diabetes130,
    get_output_dirs,
)

#main preprocessing function: load, clean, balance the raw clinical records for a dataset
#dataset: "heart_disease" (default) or "diabetes130"
def preprocessing_data(dataset="heart_disease"):
    dirs = get_output_dirs(dataset)
    sample_size = 20000
    if dataset == "diabetes130":
        datasetChoosen = load_diabetes130(sample_size=sample_size)
        target_col = "readmitted"
        num_cols_used = num_cols_diabetes130
        cat_cols_used = cat_cols_diabetes130
    else:
        datasetChoosen = load_heart_disease()
        target_col = "num"
        num_cols_used = num_cols
        cat_cols_used = cat_cols

    print(datasetChoosen.shape)
    print(datasetChoosen.head())

    # split dataset into features and target variable (the rest are features)
    X = datasetChoosen.drop(target_col, axis=1)
    y = datasetChoosen[target_col]
    y = (y > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # impute missing values in the raw feature space (required before SMOTE, and so every
    # record stays interpretable for the text description used in the embedding phase)
    X_train_imputed = impute_raw(X_train, num_cols_used, cat_cols_used)

    # balance the target class on the raw/interpretable features with SMOTENC, so synthetic
    # records keep realistic values (real scale numerics + valid categorical codes) and can be
    # converted to text just like real records — plain SMOTE only works in encoded vector space
    # and its synthetic rows can't be mapped back to a clinical record description.
    X_train_bal, y_train_bal = balance_classes(X_train_imputed, y_train, cat_cols_used)

    # Raw (pre-embedding) clinical features, row-order aligned with the embeddings generated
    # from X_train_bal, kept for traceability of predictions back to the original/synthetic record.
    X_train_bal.reset_index(drop=True).to_csv(
        os.path.join(dirs["preprocessing"], "X_train_raw.csv"), index=False
    )

    return X_train_bal.reset_index(drop=True), y_train_bal.reset_index(drop=True)


def impute_raw(X, num_cols, cat_cols):
    X = X.copy()
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode().iloc[0])
    return X

def balance_classes(X, y, cat_cols):
    # SMOTENC handles the mix of numeric and categorical columns directly on the raw
    # feature space: numeric features are interpolated, categorical ones are set to the
    # majority value among nearest neighbors, so every synthetic row is still a valid
    # (interpretable) clinical record.
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    smote_nc = SMOTENC(categorical_features=cat_idx, random_state=42)
    X_res, y_res = smote_nc.fit_resample(X, y)
    return X_res, y_res
