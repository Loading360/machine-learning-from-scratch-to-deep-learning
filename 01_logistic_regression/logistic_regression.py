
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# df = pd.read_excel("Data_emprunt.xlsx") # (penser à ajouter la base de données dans le dossier)

# Chargement du dataset Give Me Some Credit.
# Le fichier doit être placé dans le même dossier que ce script.
df = pd.read_csv(Path(__file__).resolve().parent / "cs-training.csv")

X = df.drop(columns=["Unnamed: 0", 'SeriousDlqin2yrs'], errors='ignore')  # On élimine les colonnes inutiles ou nuisibles. Ici l'identifiant des données 'Unnamed: 0' et la donnée que l'on cherche bien sûr à prédire 'SeriousDlqin2yrs'.
Y = df['SeriousDlqin2yrs']

# On doit désormais séparer le Dataset entre les données d'entraînement et de test :
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=42, stratify=Y)

Columns = df.columns

for col in Columns:
    print(type(df[col].iloc[0]))  # On vérifie quel est le type des données importées. Pour des données numériques on aura simplement à les mettre à l'échelle tandis que pour des données catégorielles on pourra envisager une méthode telle que One_hot_encoder pour traduire les données en données numériques exploitables.

# Dans ce dataset, toutes les données sont numériques

# On vérifie s'il y a des données non renseignées (NaN) dans le csv, c'est le cas ici.
print(df.isna().sum())

# On prépare désormais un "preprocessor", avec la commande "ColumnTransformer". Celle-ci s'articule sous la forme : nom, transformer, colonnes.

# Créons le transformer :
numeric_transform = make_pipeline(
    SimpleImputer(strategy='median'),     # On remplace les NaN par la médiane des valeurs de la colonne, calculée sur les données d'entraînement
    StandardScaler()                      # On centre et réduit les données
)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transform, X_train.columns.tolist())
    ]
)

# Il ne reste plus qu'à créer le modèle d'entraînement en intégrant la transformation préparatoire à faire et la fonction de classification :
# (Correction du séparateur virgule au lieu du signe égal dans la Pipeline)
model = Pipeline(
    steps=[
        ('preprocessing', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ]
)

# On peut désormais entraîner le modèle et regarder ses premières prédictions.
model.fit(X_train, y_train)
pred_train = model.predict(X_test)

print("TEST")
print("Accuracy :", accuracy_score(y_test, pred_train))              # L'accuracy définit le nombre de réponses justes
print("Recall :", recall_score(y_test, pred_train))                  # Le recall décrit la capacité d'un modèle à détecter toutes les instances positives réelles
print("Precision :", precision_score(y_test, pred_train, zero_division=0))  # La précision décrit la fiabilité des prédictions positives d'un modèle
print("F1-score :", f1_score(y_test, pred_train))                    # Le f1_score décrit la moyenne harmonique de la précision et du recall, c'est la métrique que nous chercherons à maximiser

# Désormais cherchons à optimiser les paramètres du modèle en faisant varier le paramètre de régularisation C et le seuil s de classification en le validant grâce à la validation croisée.

# On prépare les 5 batchs de validations qui serviront à tester le modèle :
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

S_values = [0.01 * i for i in range(1, 40,4)]
C_values = [0.1 * i for i in range(1, 20, 2)]

score_validation_liste = []

# Correction : le modèle ne dépend pas du seuil de classification pendant l'entraînement.
# On entraîne donc une seule fois le modèle pour chaque couple (C, fold),
# puis on teste tous les seuils sur les probabilités prédites.

for C in C_values:

    model.set_params(classifier__C=C)

    # Une liste par seuil, pour conserver les F1-scores des 5 folds.
    f1_validation_par_seuil = [[] for s in S_values]

    for train_idx, val_idx in cv.split(X_train, y_train):

        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_val_fold = y_train.iloc[val_idx]

        model.fit(X_train_fold, y_train_fold)

        # On calcule une seule fois les probabilités pour ce fold.
        proba_val = model.predict_proba(X_val_fold)[:, 1]

        for j, s in enumerate(S_values):

            pred_val = (proba_val > s).astype(int)
            score_val = f1_score(y_val_fold, pred_val, zero_division=0)

            f1_validation_par_seuil[j].append(score_val)

    # On calcule la moyenne des F1-scores obtenus sur les 5 folds
    # pour chaque seuil de classification.
    score_validation_liste_sous_liste = [
        np.mean(scores) for scores in f1_validation_par_seuil
    ]

    score_validation_liste.append(score_validation_liste_sous_liste)

# Conversion de la liste de listes en tableau numpy pour la heatmap
score_matrix = np.array(score_validation_liste)

plt.figure(figsize=(10, 6))

sns.heatmap(
    score_matrix,
    annot=True,
    fmt=".3f",
    xticklabels=[f"{s:.2f}" for s in S_values],
    yticklabels=[f"{C:.1f}" for C in C_values],
    cmap="viridis"
)

plt.xlabel("Seuil")
plt.ylabel("C")
plt.title("Mean cross-validation F1-score")
plt.tight_layout()
plt.show()

# On récupère automatiquement les indices du meilleur couple (C, seuil).
# np.argmax donne l'indice du plus grand F1-score dans la matrice.
best_C_idx, best_S_idx = np.unravel_index(
    np.argmax(score_matrix),
    score_matrix.shape
)

best_C = C_values[best_C_idx]
best_s = S_values[best_S_idx]
best_f1_cv = score_matrix[best_C_idx, best_S_idx]

print("\nBEST HYPERPARAMETERS")
print("Best C :", best_C)
print("Best threshold :", best_s)
print("Mean CV F1-score :", best_f1_cv)

# Maintenant que les hyperparamètres ont été sélectionnés grâce à la validation
# croisée, on réentraîne le modèle sur l'intégralité du jeu d'entraînement.
model.set_params(classifier__C=best_C)
model.fit(X_train, y_train)

# On applique le seuil optimal aux probabilités prédites sur le jeu de test.
proba_test = model.predict_proba(X_test)[:, 1]
pred_test = (proba_test > best_s).astype(int)

# Le jeu de test n'a pas été utilisé pour choisir C ou le seuil.
# Il fournit donc une évaluation finale indépendante de l'optimisation.
print("\nFINAL TEST RESULTS")
print("Accuracy :", accuracy_score(y_test, pred_test))
print("Recall :", recall_score(y_test, pred_test))
print("Precision :", precision_score(y_test, pred_test, zero_division=0))
print("F1-score :", f1_score(y_test, pred_test))
