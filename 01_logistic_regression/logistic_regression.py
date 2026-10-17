"""Part I — Logistic regression for credit-default prediction.

Dataset: Kaggle's Give Me Some Credit (cs-training.csv).
The model and decision threshold are selected using stratified cross-validation.
The held-out test set is evaluated only after hyperparameter selection.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


# Load the Give Me Some Credit dataset.
# Prefer a shared data/ directory at the repository root, but also support
# a CSV placed next to this script for convenient local execution.
script_dir = Path(__file__).resolve().parent
data_candidates = [
    script_dir.parent / "data" / "cs-training.csv",
    script_dir / "cs-training.csv",
]
data_path = next((path for path in data_candidates if path.is_file()), None)
if data_path is None:
    raise FileNotFoundError(
        "Download cs-training.csv from the Give Me Some Credit Kaggle competition "
        "and place it in the repository's data/ directory or beside this script."
    )
df = pd.read_csv(data_path)

# Remove the imported row identifier ('Unnamed: 0') and the target variable
# ('SeriousDlqin2yrs') from the explanatory variables.
X = df.drop(columns=["Unnamed: 0", "SeriousDlqin2yrs"], errors="ignore")
Y = df["SeriousDlqin2yrs"]

# Split the dataset into training and test sets.
# Stratification preserves the minority-class proportion in both subsets.
# The test set must remain untouched while selecting hyperparameters.
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=42, stratify=Y
)

Columns = df.columns

# Inspect the Python/NumPy types of the imported columns.
# Numerical variables can be scaled directly; categorical variables would
# instead require an encoding procedure such as OneHotEncoder.
for col in Columns:
    print(type(df[col].iloc[0]))

# All explanatory variables in this dataset are numerical.

# Check the number of missing values (NaN) in each column.
# MonthlyIncome and NumberOfDependents contain missing observations.
print(df.isna().sum())

# Prepare a ColumnTransformer: (name, transformer, columns).
# Preprocessing is fitted inside each training fold to prevent data leakage.

# Create the numerical transformation pipeline.
numeric_transform = make_pipeline(
    SimpleImputer(strategy="median"),  # Replace NaNs with training-fold medians.
    StandardScaler(),                  # Center and scale numerical features.
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transform, X_train.columns.tolist())
    ]
)

# Combine preprocessing and logistic regression into a single Pipeline.
# This ensures the same transformations are applied at training and prediction.
model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# Tune the regularization parameter C and the probability threshold s.
# C is the inverse regularization strength: smaller C means stronger penalty.
# The threshold controls the trade-off between precision and recall.

# Create five stratified validation folds.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

S_values = [0.01 * i for i in range(1, 40, 4)]
C_values = [0.1 * i for i in range(1, 20, 2)]

score_validation_liste = []

# The classification threshold does not affect model fitting.
# Therefore, fit only once per (C, fold), then evaluate all thresholds
# using the same predicted probabilities. This avoids redundant training.
for C in C_values:
    model.set_params(classifier__C=C)

    # Store the five fold-specific F1-scores for each candidate threshold.
    f1_validation_par_seuil = [[] for s in S_values]

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_val_fold = y_train.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)

        # Predict positive-class probabilities once per validation fold.
        proba_val = model.predict_proba(X_val_fold)[:, 1]

        for j, s in enumerate(S_values):
            pred_val = (proba_val > s).astype(int)
            score_val = f1_score(y_val_fold, pred_val, zero_division=0)
            f1_validation_par_seuil[j].append(score_val)

    # Average the F1-score across the five validation folds for each threshold.
    score_validation_liste_sous_liste = [
        np.mean(scores) for scores in f1_validation_par_seuil
    ]
    score_validation_liste.append(score_validation_liste_sous_liste)

# Convert the nested lists to a matrix for the heatmap.
score_matrix = np.array(score_validation_liste)

plt.figure(figsize=(10, 6))
sns.heatmap(
    score_matrix,
    annot=True,
    fmt=".3f",
    xticklabels=[f"{s:.2f}" for s in S_values],
    yticklabels=[f"{C:.1f}" for C in C_values],
    cmap="viridis",
)
plt.xlabel("Decision threshold")
plt.ylabel("Regularization parameter C")
plt.title("Mean cross-validation F1-score")
plt.tight_layout()

# Save a reproducible version of the heatmap in the figures directory.
figures_dir = script_dir / "figures"
figures_dir.mkdir(exist_ok=True)
plt.savefig(figures_dir / "logistic_cv_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()

# Retrieve the indices of the (C, threshold) pair maximizing mean CV F1.
# np.argmax returns the maximum's flattened index; unravel_index converts
# it back to the corresponding row and column.
best_C_idx, best_S_idx = np.unravel_index(
    np.argmax(score_matrix), score_matrix.shape
)

best_C = C_values[best_C_idx]
best_s = S_values[best_S_idx]
best_f1_cv = score_matrix[best_C_idx, best_S_idx]

print("\nBEST HYPERPARAMETERS")
print("Best C :", best_C)
print("Best threshold :", best_s)
print("Mean CV F1-score :", best_f1_cv)

# Fit the selected model on the full training set, now that cross-validation
# has determined the hyperparameters.
model.set_params(classifier__C=best_C)
model.fit(X_train, y_train)

# Evaluate the baseline (C=1, threshold=0.5) on the held-out test set.
# We use a separate pipeline so the selected model remains unchanged.
baseline_model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(C=1.0, max_iter=1000)),
    ]
)
baseline_model.fit(X_train, y_train)
pred_baseline = baseline_model.predict(X_test)

print("\nBASELINE TEST RESULTS (C=1, threshold=0.5)")
print("Accuracy :", accuracy_score(y_test, pred_baseline))
print("Recall :", recall_score(y_test, pred_baseline))
print("Precision :", precision_score(y_test, pred_baseline, zero_division=0))
print("F1-score :", f1_score(y_test, pred_baseline, zero_division=0))

# Apply the selected threshold to the positive-class probabilities on X_test.
proba_test = model.predict_proba(X_test)[:, 1]
pred_test = (proba_test > best_s).astype(int)

# The test set was not used to choose C or the threshold.
# It therefore provides a held-out evaluation of the selected configuration.
print("\nFINAL TEST RESULTS")
print("Accuracy :", accuracy_score(y_test, pred_test))
print("Recall :", recall_score(y_test, pred_test))
print("Precision :", precision_score(y_test, pred_test, zero_division=0))
print("F1-score :", f1_score(y_test, pred_test, zero_division=0))
