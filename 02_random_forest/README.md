# Part II — Random Forest

## Research question

Can an ensemble of nonlinear decision trees predict serious credit delinquency more effectively than the logistic regression benchmark from [Part I](../01_logistic_regression/)?

This section uses the **same Give Me Some Credit dataset** and the **same stratified 80/20 train/test split** as Part I, allowing a direct comparison of held-out classification performance.

## Dataset and preprocessing

- **Source:** [Kaggle — Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data)
- **Observations:** approximately 150,000; 10 numerical predictors
- **Target:** `SeriousDlqin2yrs` (serious delinquency within two years)
- **Positive class:** approximately 6.7%, making accuracy alone a misleading metric
- **Missing values:** median imputation, fitted inside each cross-validation training fold

Unlike logistic regression, Random Forest does **not require feature standardization**. The scikit-learn `Pipeline` keeps imputation within cross-validation to avoid data leakage.

## Model and hyperparameter search

The model is a `RandomForestClassifier` with **100 trees**, `max_features="sqrt"`, bootstrap sampling, and `random_state=42`.

A **five-fold stratified cross-validation** on the training set jointly searches:

| Hyperparameter | Candidate values |
| --- | --- |
| `max_depth` | 2, 4, 6, 8, None |
| `min_samples_leaf` | 1, 2, 5, 10, 20, 50 |
| Decision threshold | 0.01, 0.05, 0.09, 0.13, 0.17, 0.21, 0.25, 0.29, 0.33, 0.37 |

The selection criterion is **mean validation F1-score**. Predicted probabilities are calculated once per model/fold and reused across all thresholds. This requires **150 forest fits** instead of 1,500.

## Cross-validation findings

The author's reported run selected:

- **Maximum depth:** 8
- **Minimum samples per leaf:** 1
- **Decision threshold:** 0.21

The heatmaps suggest that trees with depth 2 are too constrained, while deeper trees capture more useful nonlinear relationships. With unlimited depth, very small leaves tend to perform less well, consistent with increased variance. The score is often relatively insensitive to leaf size when depth is restricted.

**Heatmap figure:** The script generates `figures/random_forest_cv_heatmaps.png` automatically. Upload that output here to display it in the README. The original heatmap has not yet been added to the repository.

## Held-out test results

| Metric | Initial forest (depth=3, leaf=5, threshold=0.5) | Tuned Random Forest | Tuned Logistic Regression |
| --- | ---: | ---: | ---: |
| Accuracy | 0.9344 | **0.9189** | 0.9139 |
| Recall | 0.0334 | **0.4793** | 0.3162 |
| Precision | 0.6837 | **0.4091** | 0.3433 |
| **F1-score** | **0.0637** | **0.4414** | 0.3292 |

*Results reported from the author's executions, not independently rerun here. The initial forest and tuned forest use different classification thresholds. The logistic regression benchmark comes from Part I.*

The tuned Random Forest achieves a test F1-score of **0.4414**, versus **0.3292** for logistic regression: an improvement of approximately **34% relative to the logistic regression F1-score**. Both recall and precision are higher for the tuned forest, although the high baseline accuracy reflects the large majority of negative cases.

## Interpretation

Random Forest can capture nonlinear effects and interactions that a basic logistic regression cannot represent without explicit feature engineering. For credit-risk data, relationships such as delinquency history, credit utilization, and income may involve thresholds and interactions.

The decision threshold is essential: lowering it from 0.5 increases the proportion of positive cases detected, but also creates more false positives. Maximizing F1 is useful for comparing models, **not a substitute for a cost-sensitive credit decision rule**.

These results are exploratory. The test set has now been inspected for both model families, so further algorithm selection based on their test scores would risk optimistic model comparison. Future experiments should rely on training-only cross-validation or a new untouched holdout.

## Reproduce

Install the required Python packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Download `cs-training.csv` from the Kaggle competition and place it in `data/` at the repository root (or beside this script). Run:

```bash
python 02_random_forest/Random_Forest.py
```

The script prints selected hyperparameters and classification metrics, and saves five cross-validation heatmaps in `02_random_forest/figures/`.

## Limitations and next steps

- Hyperparameters are selected on a finite grid, not a continuous search.
- `n_estimators=100` and `max_features="sqrt"` are fixed rather than tuned.
- Mean F1-score is the optimization target; no explicit monetary cost matrix is specified.
- Probability calibration, uncertainty in cross-validation scores, and further nonlinear baselines remain to be investigated.
- Part III will investigate multilayer perceptrons on the same dataset.
