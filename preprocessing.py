import os

from sklearn.model_selection import train_test_split

from function import load_heart_disease, load_diabetes130, get_output_dirs

#main preprocessing function: load the raw clinical records and hold out an untouched test set
#dataset: "heart_disease" (default) or "diabetes130"
def preprocessing_data(dataset="heart_disease"):
    dirs = get_output_dirs(dataset)
    sample_size = 20000
    if dataset == "diabetes130":
        datasetChoosen = load_diabetes130(sample_size=sample_size)
        target_col = "readmitted"
    else:
        datasetChoosen = load_heart_disease()
        target_col = "num"

    print(datasetChoosen.shape)
    print(datasetChoosen.head())

    # split dataset into features and target variable (the rest are features)
    X = datasetChoosen.drop(target_col, axis=1)
    y = datasetChoosen[target_col]
    y = (y > 0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # No imputation and no oversampling: a missing value stays missing and becomes "not recorded"
    # in the text description, exactly like a "Non lo so" answer in chat. Oversampling before the
    # cross-validation taught the classifier to spot synthetic records instead of the clinical task.
    X_train.reset_index(drop=True).to_csv(
        os.path.join(dirs["preprocessing"], "X_train_raw.csv"), index=False
    )

    return (
        X_train.reset_index(drop=True), X_test.reset_index(drop=True),
        y_train.reset_index(drop=True), y_test.reset_index(drop=True),
    )
