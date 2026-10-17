from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# On conserve le même jeux de données traités que précédemment, le code change à partir du Pipeline

model = Pipeline(
  steps = [
        ('preprocessing', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators = 100, 
                                              max_depth = 3, 
                                              max_features = "sqrt", 
                                              bootstrap = True, 
                                              random_state = 42, 
                                              min_samples_leaf = 5))
    ]
)

model.fit(X_train, y_train)
pred_train = model.predict(X_test)

print("TEST")
print("Accuracy :", accuracy_score(y_test, pred_train))             
print("Recall :", recall_score(y_test, pred_train))                
print("Precision :", precision_score(y_test, pred_train, zero_division=0)) 
print("F1-score :", f1_score(y_test, pred_train))                  









# On prépare les 5 batchs de validations qui serviront à tester le modèle :
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

max_depth_values = [2,4,6,8, None]
min_samples_leaf_values = [1, 2, 5, 10, 20, 50]
S_values = [0.01 * i for i in range(1, 40,4)]


score_validation_liste = []



for l in min_samples_leaf_values:
    score_validation_liste_par_profondeur = []
    for m in max_depth_values :
        model = Pipeline(
            steps = [
                    ('preprocessing', preprocessor),
                    ('classifier', RandomForestClassifier(n_estimators = 100, 
                                                        max_depth = m, 
                                                        max_features = "sqrt", 
                                                        bootstrap = True, 
                                                        random_state = 42, 
                                                        min_samples_leaf = l))
                ]
            )
            
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
    
        score_validation_liste_par_profondeur.append(score_validation_liste_sous_liste)
    score_validation_liste.append(score_validation_liste_par_profondeur)

# Conversion de la liste de listes en tableau numpy pour la heatmap
score_matrix = np.array(score_validation_liste)



fig, axes = plt.subplots(1, len(max_depth_values),
                         figsize=(22, 5), sharey=True)

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
        cbar=(i == len(max_depth_values) - 1)
    )

    axes[i].set_title(f"Max depth = {depth}")
    axes[i].set_xlabel("Threshold")

axes[0].set_ylabel("Min samples leaf")
plt.tight_layout()
plt.show()



# np.argmax donne l'indice du plus grand F1-score dans la matrice.
best_sample_idx, best_depth_idx, best_S_idx = np.unravel_index(
    np.argmax(score_matrix),
    score_matrix.shape
)

best_sample = min_samples_leaf_values[best_sample_idx]
best_depth = max_depth_values[best_depth_idx]
best_s = S_values[best_S_idx]

print("\nBEST HYPERPARAMETERS")
print("Best Depth :", best_depth)
print("Best sample :", best_sample)
print("Best threshold :", best_s)

# Maintenant que les hyperparamètres ont été sélectionnés grâce à la validation
# croisée, on réentraîne le modèle sur l'intégralité du jeu d'entraînement.
model = Pipeline(
    steps = [
            ('preprocessing', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators = 100, 
                                                max_depth = best_depth, 
                                                max_features = "sqrt", 
                                                bootstrap = True, 
                                                random_state = 42, 
                                                min_samples_leaf = best_sample))
        ]
    )
model.fit(X_train, y_train)

# On applique le seuil optimal aux probabilités prédites sur le jeu de test.
proba_test = model.predict_proba(X_test)[:, 1]
pred_test = (proba_test > best_s).astype(int)

# Il fournit donc une évaluation finale indépendante de l'optimisation.
print("\nFINAL TEST RESULTS")
print("Accuracy :", accuracy_score(y_test, pred_test))
print("Recall :", recall_score(y_test, pred_test))
print("Precision :", precision_score(y_test, pred_test, zero_division=0))
print("F1-score :", f1_score(y_test, pred_test))









