# Part III — Multilayer Perceptron (MLP)

## Objective

Predict serious delinquency within two years using a feed-forward neural network, and compare its performance with the logistic regression and random forest models in Parts I and II. The task is **imbalanced binary classification**: approximately 6.7% of observations belong to the positive class. Consequently, accuracy alone is not an informative measure of model quality; the principal selection metric is the **F1-score**.

**Dataset:** [Kaggle — Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit/data), file `cs-training.csv` (approximately 150,000 observations, 10 numerical predictors). The dataset is not redistributed here.

## Model and preprocessing

The experiment uses scikit-learn's `MLPClassifier`:

- **Architecture:** one hidden layer with 16 neurons and ReLU activation.
- **Optimization:** Adam, learning rate 0.001, maximum 1,000 iterations.
- **Regularization:** L2 penalty controlled by `alpha`.
- **Preprocessing:** median imputation of missing numerical values, followed by standardization.
- **Reproducibility:** `random_state=42`.

Preprocessing is included in a `Pipeline` so that the imputer and scaler are fitted **only on the training portion of each cross-validation fold**, preventing leakage from validation data.

## Experimental protocol

1. Hold out **20%** of the dataset for final testing, using a stratified 80/20 split.
2. On the remaining 80%, use **5-fold stratified cross-validation**.
3. Jointly search 20 L2 regularization values (from **0.0005 to 0.048**, step 0.0025) and 10 probability thresholds (from **0.02 to 0.38**, step 0.04).
4. For each `alpha` and CV fold, fit the network once, then evaluate all 10 thresholds on the validation probabilities. This requires **100 network fits**, not 1,000.
5. Choose the pair maximizing mean validation F1; retrain on the entire training split and evaluate once on the held-out test set.

The decision threshold is tuned separately from the model's parameters: lowering it increases the number of predicted positive cases, generally improving recall while reducing precision.

## Results

**Best cross-validation configuration:**

| Hyperparameter | Selected value |
|---|---:|
| L2 regularization `alpha` | **0.005** |
| Decision threshold | **0.18** |
| Mean 5-fold CV F1-score | **0.42495** |

**Held-out test set:**

| Metric | Baseline (alpha=1, threshold=0.50) | CV-optimized (alpha=0.005, threshold=0.18) |
|---|---:|---:|
| Accuracy | 0.93587 | 0.91140 |
| Recall | 0.09925 | **0.49925** |
| Precision | **0.62776** | 0.37702 |
| F1-score | 0.17140 | **0.42961** |

Optimizing the classification threshold greatly improves minority-class detection: recall increases from approximately **9.9% to 49.9%**. This comes at the expense of precision and accuracy, a meaningful trade-off in this imbalanced problem.

### Cross-validation heatmap

The script generates `figures/mlp_cv_heatmap.png`, with mean validation F1 across regularization values and decision thresholds. The heatmap indicates that **threshold choice matters substantially more than the small changes in alpha explored here**. The best scores are concentrated around thresholds 0.14–0.22, with only modest variation across the tested regularization strengths.

Once the figure is added to the repository, it can be viewed at `figures/mlp_cv_heatmap.png`.

## Comparison with earlier parts

| Model | Optimized test F1-score |
|---|---:|
| Part I — Logistic regression | 0.32918 |
| Part II — Random forest | **0.44143** |
| Part III — MLP | 0.42961 |

The MLP substantially improves upon the logistic regression but does **not** outperform the random forest on this dataset. This is consistent with the observation that deep learning is not automatically superior on relatively small tabular feature spaces.

**Interpretation caution:** These are descriptive held-out results for three experiments, not a statistical significance test of model differences. Model selection should rely on cross-validation, rather than repeatedly optimizing choices against the test set.

## Reproduce

Install the dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

Download `cs-training.csv` from Kaggle and put it in the repository's `data/` folder, next to `mlp.py`, or in the current working directory. Then run:

```bash
python 03_multilayer_perceptron/mlp.py
```

The full CV experiment may take a substantial amount of time on a CPU. The script prints the selected hyperparameters and test metrics and saves the heatmap to `03_multilayer_perceptron/figures/mlp_cv_heatmap.png`.

## Limitations and possible extensions

- Only a **single 16-neuron hidden layer** was investigated; the experiment does not optimize network depth or width.
- The tested `alpha` range is narrow; a logarithmic grid could explore stronger regularization.
- The reported F1-optimal threshold does not incorporate the asymmetric economic costs of false positives and false negatives.
- A dedicated independent evaluation dataset would be useful before making claims about deployment performance.
