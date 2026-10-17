import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# df = pd.read_excel("Data_emprunt.xlsx") # (penser à ajouter la base de données dans le dossier)

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
    SimpleImputer(strategy='median'),     # On supprime les NaN en injectant la médiane des valeurs de la colonne
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
print("Precision :", precision_score(y_test, pred_train))            # La précision décrit la fiabilité des prédictions positives d'un modèle
print("F1-score :", f1_score(y_test, pred_train))                    # Le f1_score décrit la moyenne harmonique de la précision et du recall, c'est la métrique que nous chercherons à maximiser

# Désormais cherchons à optimiser les paramètres du modèle en faisant varier le paramètre de régularisation C et le seuil s de classification en le validant grâce à la validation croisée. 

# On prépare les 5 batchs de validations qui serviront à tester le modèle : 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

S_values = [0.05 * i for i in range(1, 10,2)]
C_values = [0.1 * i for i in range(1, 40, 5)]

score_validation_liste = []

for C in C_values:
    score_validation_liste_sous_liste = []
    for s in S_values:
        
        model.set_params(classifier__C=C)
        f1_validation = []
        for train_idx, val_idx in cv.split(X_train, y_train):
        
            X_train_fold = X_train.iloc[train_idx]
            y_train_fold = y_train.iloc[train_idx]
            X_val_fold = X_train.iloc[val_idx]
            y_val_fold = y_train.iloc[val_idx]
        
            model.fit(X_train_fold, y_train_fold)

            pred_val = (model.predict_proba(X_val_fold)[:, 1] > s).astype(int)
            score_val = f1_score(y_val_fold, pred_val)

            f1_validation.append(score_val)
    
        score_val = np.mean(f1_validation)
        score_validation_liste_sous_liste.append(score_val)
        
    score_validation_liste.append(score_validation_liste_sous_liste)

# Conversion de la liste de listes en tableau numpy pour la heatmap
score_matrix = np.array(score_validation_liste)

sns.heatmap(
    score_matrix,
    annot=True,
    fmt=".2f",
    xticklabels=S_values,
    yticklabels=C_values
)

plt.xlabel("Seuil")
plt.ylabel("C")
plt.show()
