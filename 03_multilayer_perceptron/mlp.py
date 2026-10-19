"""Part III — Multilayer Perceptron on the Give Me Some Credit dataset.

A single-hidden-layer neural network trained with Adam. The L2 regularization
parameter and probability decision threshold are selected jointly through
5-fold stratified cross-validation on the training split only.

Dataset: Kaggle 'Give Me Some Credit', cs-training.csv.
Run: python mlp.py
The CSV can be placed in ../data/, in this directory, or in the working directory.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
CSV_CANDIDATES = (
    SCRIPT_DIR.parent / "data" / "cs-training.csv",
    SCRIPT_DIR / "cs-training.csv",
    Path.cwd() / "cs-training.csv",
)


def load_data():
    csv_path = next((p for p in CSV_CANDIDATES if p.is_file()), None)
    if csv_path is None:
        raise FileNotFoundError(
            "Download cs-training.csv from Kaggle 'Give Me Some Credit' "
            "and place it in data/, next to this script, or in the working directory."
        )
    data = pd.read_csv(csv_path)
    x = data.drop(columns=["Unnamed: 0", "SeriousDlqin2yrs"], errors="ignore")
    y = data["SeriousDlqin2yrs"]
    return x, y


def make_model(columns, alpha=1.0):
    # Imputation and scaling are fitted separately within each CV training fold:
    # this avoids leaking validation statistics into training.
    numeric_transform = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transform, list(columns))]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("classifier", MLPClassifier(
                hidden_layer_sizes=(16,),
                activation="relu",
                solver="adam",
                alpha=alpha,
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=42,
            )),
        ]
    )


def report_metrics(name, y_true, predictions):
    print(f"\n{name}")
    print("Accuracy :", accuracy_score(y_true, predictions))
    print("Recall :", recall_score(y_true, predictions, zero_division=0))
    print("Precision :", precision_score(y_true, predictions, zero_division=0))
    print("F1-score :", f1_score(y_true, predictions, zero_division=0))


def main():
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, train_size=0.8, test_size=0.2,
        random_state=42, stratify=y,
    )

    # Original experiment: 20 alpha values x 5 folds = 100 model fits.
    # Each fitted model is evaluated at 10 thresholds without retraining.
    thresholds = [0.01 * i for i in range(2, 40, 4)]
    alpha_values = [0.0005 * i for i in range(1, 100, 5)]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score_matrix = np.zeros((len(alpha_values), len(thresholds)))

    for i, alpha in enumerate(alpha_values):
        fold_scores = [[] for _ in thresholds]
        for train_idx, val_idx in cv.split(x_train, y_train):
            x_fold_train = x_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            x_fold_val = x_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]

            model = make_model(x_train.columns, alpha=alpha)
            model.fit(x_fold_train, y_fold_train)
            probabilities = model.predict_proba(x_fold_val)[:, 1]

            for j, threshold in enumerate(thresholds):
                predictions = (probabilities > threshold).astype(int)
                fold_scores[j].append(
                    f1_score(y_fold_val, predictions, zero_division=0)
                )

        score_matrix[i] = [np.mean(scores) for scores in fold_scores]

    # Keep enough decimal places to distinguish the tested alpha values.
    plt.figure(figsize=(11, 8))
    sns.heatmap(
        score_matrix, annot=True, fmt=".3f",
        xticklabels=[f"{s:.2f}" for s in thresholds],
        yticklabels=[f"{a:.4f}" for a in alpha_values],
        cmap="viridis",
    )
    plt.xlabel("Decision threshold")
    plt.ylabel("Regularization parameter alpha")
    plt.title("Mean cross-validation F1-score")
    plt.tight_layout()
    figures_dir = SCRIPT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / "mlp_cv_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

    best_alpha_idx, best_threshold_idx = np.unravel_index(
        np.argmax(score_matrix), score_matrix.shape
    )
    best_alpha = alpha_values[best_alpha_idx]
    best_threshold = thresholds[best_threshold_idx]
    best_cv_f1 = score_matrix[best_alpha_idx, best_threshold_idx]

    print("\nBEST HYPERPARAMETERS")
    print("Best alpha :", best_alpha)
    print("Best threshold :", best_threshold)
    print("Mean CV F1-score :", best_cv_f1)

    # Baseline: original fixed regularization and standard 0.5 cutoff.
    baseline_model = make_model(x_train.columns, alpha=1.0)
    baseline_model.fit(x_train, y_train)
    baseline_predictions = baseline_model.predict(x_test)
    report_metrics(
        "BASELINE TEST RESULTS (alpha=1, threshold=0.5)",
        y_test, baseline_predictions,
    )

    # Final evaluation: CV-selected configuration, fitted on full training set.
    final_model = make_model(x_train.columns, alpha=best_alpha)
    final_model.fit(x_train, y_train)
    test_probabilities = final_model.predict_proba(x_test)[:, 1]
    final_predictions = (test_probabilities > best_threshold).astype(int)
    report_metrics("FINAL TEST RESULTS", y_test, final_predictions)


if __name__ == "__main__":
    main()
