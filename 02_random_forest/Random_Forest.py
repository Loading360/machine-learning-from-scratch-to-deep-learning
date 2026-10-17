"""Part II — Random Forest classification on the Give Me Some Credit dataset.

Five-fold stratified cross-validation jointly selects maximum tree depth,
minimum leaf size, and the classification threshold using mean F1-score.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline


# Use the same dataset and train/test split as in Part I (logistic regression).
# Unlike linear models, decision trees do not need feature standardization.
# We keep median imputation, fitted separately within each training fold.
script_dir = Path(__file__).resolve().parent
data_candidates = [
    script_dir.parent / "data" / "cs-training.csv",
    script_dir / "cs-training.csv",
]
data_path = next((path for path in data_candidates if path.is_file()), None)
if data_path is None:
    raise FileNotFoundError(
        "Download cs-training.csv from Kaggle's Give Me Some Credit competition "
        "and place it in the repository's data/ directory or beside this script."
    )

df = pd.read_csv(data_path)
X = df.drop(columns=["Unnamed: 0", "SeriousDlqin2yrs"], errors="ignore")
Y = df["SeriousDlqin2yrs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, train_size=0.8, test_size=0.2, random_state=42, stratify=Y
)


# Construct a pipeline so missing-value imputation is learned only from
# the training data in each fold, avoiding information leakage.
def make_forest(max_depth, min_samples_leaf):
    return Pipeline(
        steps=[
            ("preprocessing", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=100,       # Number of trees in the ensemble.
                max_depth=max_depth,   # Maximum depth of each tree (None = unlimited).
                max_features="sqrt",   # Random feature subset at each split.
                bootstrap=True,        # Fit trees on bootstrap samples.
                random_state=42,       # Reproducible random sampling.
                min_samples_leaf=min_samples_leaf,
                n_jobs=-1,             # Parallelize tree construction.
            )),
        ]
    )


# Prepare five stratified folds for model selection.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

max_depth_values = [2, 4, 6, 8, None]
min_samples_leaf_values = [1, 2, 5, 10, 20, 50]
S_values = [0.01 * i for i in range(1, 40, 4)]

score_validation_liste = []

# Explore the minimum leaf size and maximum depth.
# For each model and fold, predict probabilities only once and then test
# all thresholds without retraining. This saves considerable computation.
for l in min_samples_leaf_values:
    score_validation_liste_par_profondeur = []

    for m in max_depth_values:
        model = make_forest(max_depth=m, min_samples_leaf=l)

        # One list per threshold, containing the five validation F1-scores.
        f1_validation_par_seuil = [[] for s in S_values]

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]

            model.fit(X_train_fold, y_train_fold)

            # Compute positive-class probabilities once for this fold.
            proba_val = model.predict_proba(X_val_fold)[:, 1]

            for j, s in enumerate(S_values):
                pred_val = (proba_val > s).astype(int)
                score_val = f1_score(y_val_fold, pred_val, zero_division=0)
                f1_validation_par_seuil[j].append(score_val)

        # Average F1-scores across the five folds for each threshold.
        score_validation_liste_sous_liste = [
            np.mean(scores) for scores in f1_validation_par_seuil
        ]
        score_validation_liste_par_profondeur.append(
            score_validation_liste_sous_liste
        )

    score_validation_liste.append(score_validation_liste_par_profondeur)


# Convert the nested lists into a 3D array:
# (minimum leaf size, maximum depth, decision threshold).
score_matrix = np.array(score_validation_liste)

# A single heatmap cannot represent a 3D score matrix.
# Plot one heatmap per maximum depth, with a shared color scale.
fig, axes = plt.subplots(
    1, len(max_depth_values), figsize=(22, 5), sharey=True
)

for i, depth in enumerate(max_depth_values):
    sns.heatmap(
        score_matrix[:, i, :],
        ax=axes[i],
        annot=True,
        fmt=".2f",
        xticklabels=[f"{s:.2f}" for s in S_values],
        yticklabels=min_samples_leaf_values,
        cmap="viridis",
        vmin=score_matrix.min(),
        vmax=score_matrix.max(),
        cbar=(i == len(max_depth_values) - 1),
    )
    axes[i].set_title(f"Max depth = {depth}")
    axes[i].set_xlabel("Decision threshold")

axes[0].set_ylabel("Minimum samples per leaf")
plt.tight_layout()

# Save the figure for the GitHub README.
figures_dir = script_dir / "figures"
figures_dir.mkdir(exist_ok=True)
plt.savefig(
    figures_dir / "random_forest_cv_heatmaps.png",
    dpi=200,
    bbox_inches="tight",
)
plt.show()


# np.argmax returns the position of the highest cross-validation F1-score.
# Convert the 3D indices back to actual hyperparameter values.
best_sample_idx, best_depth_idx, best_S_idx = np.unravel_index(
    np.argmax(score_matrix), score_matrix.shape
)

best_sample = min_samples_leaf_values[best_sample_idx]
best_depth = max_depth_values[best_depth_idx]
best_s = S_values[best_S_idx]
best_f1_cv = score_matrix[best_sample_idx, best_depth_idx, best_S_idx]

print("\nBEST HYPERPARAMETERS")
print("Best depth :", best_depth)
print("Best minimum samples per leaf :", best_sample)
print("Best threshold :", best_s)
print("Mean CV F1-score :", best_f1_cv)


# Once hyperparameters are selected using cross-validation, retrain on
# the entire training set. Keep X_test out of model selection.
model = make_forest(max_depth=best_depth, min_samples_leaf=best_sample)
model.fit(X_train, y_train)

# For comparison, evaluate the original untuned configuration on the same
# held-out test set, but only after model selection has been completed.
baseline_model = make_forest(max_depth=3, min_samples_leaf=5)
baseline_model.fit(X_train, y_train)
pred_baseline = baseline_model.predict(X_test)

print("\nBASELINE TEST RESULTS (depth=3, min_samples_leaf=5, threshold=0.5)")
print("Accuracy :", accuracy_score(y_test, pred_baseline))
print("Recall :", recall_score(y_test, pred_baseline))
print("Precision :", precision_score(y_test, pred_baseline, zero_division=0))
print("F1-score :", f1_score(y_test, pred_baseline, zero_division=0))

# Apply the selected probability threshold to the held-out test predictions.
proba_test = model.predict_proba(X_test)[:, 1]
pred_test = (proba_test > best_s).astype(int)

# Final held-out evaluation. The test set was not used for tuning.
print("\nFINAL TEST RESULTS")
print("Accuracy :", accuracy_score(y_test, pred_test))
print("Recall :", recall_score(y_test, pred_test))
print("Precision :", precision_score(y_test, pred_test, zero_division=0))
print("F1-score :", f1_score(y_test, pred_test, zero_division=0))
