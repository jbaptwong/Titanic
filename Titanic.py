# -*- coding: utf-8 -*-
"""
V1 de la subsmission pour le challenge Titanic Kaggle
"""

# -------------------------------
# Importation des librairies
# -------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from ydata_profiling import ProfileReport

#librairie de visualisation de l'évaluation des modèles
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix, ROCAUC

import os

# -------------------------------
# Paramétrages
# -------------------------------

path = "G:\\Mon Drive\\Kaggle\\Titanic"

os.chdir(path)

# -------------------------------
# lecture des fichiers
# -------------------------------

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data.info()
train_data.describe()

# Préparation des données
# Remplir les valeurs manquantes pour 'Age' avec la médiane
train_data['Age'].fillna(train_data['Age'].median(), inplace=True)

# Remplir les valeurs manquantes pour 'Fare' avec la médiane
train_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)

# Encoder 'Sex' en numérique (0 = male, 1 = female)
train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})

# Choisir les caractéristiques de la baseline (features)
features = ['Pclass', 'Sex', 'Age', 'Fare']

# Séparer les caractéristiques (X) et la variable cible (y)
X = train_data[features]
y = train_data['Survived']

# Séparer les données en jeu d'entraînement et de validation# Séparer les données en jeu d'entraînement et de validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle de régression logistique
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Prédire sur le jeu de validation
y_pred = model.predict(X_val)

# -------------------------------
# Visualisation de la précision du modèle
# -------------------------------

# Calculer la précision de la baseline
accuracy = accuracy_score(y_val, y_pred)
print(f'Baseline Accuracy: {accuracy:.2f}')

cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:\n", cm)

# Afficher un rapport plus complet
print("Classification Report:\n", classification_report(y_val, y_pred))

# Visualisation d'un rapport de classification
visualizer = ClassificationReport(model, support=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_val, y_val)
visualizer.show()

# Visualiser la matrice de confusion
cm = ConfusionMatrix(model)
cm.fit(X_train, y_train)
cm.score(X_val, y_val)
cm.show()

# Visualiser la courbe ROC-AUC
roc = ROCAUC(model)
roc.fit(X_train, y_train)
roc.score(X_val, y_val)
roc.show()

"""
Préparation du modèle à soumettre
"""

# -------------------------------
# Recalibrage sur la totalité des observations train
# -------------------------------

model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Appliquer les mêmes transformations que sur le jeu d'entraînement
test_data['Age'].fillna(train_data['Age'].median(), inplace=True)
test_data['Fare'].fillna(train_data['Fare'].median(), inplace=True)
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})

# Sélectionner les mêmes features utilisées pour l'entraînement
X_test = test_data[features]  # 'features' est la liste des variables que tu as utilisées pour le modèle

# Prédire les survies sur les données de test
predictions = model.predict(X_test)

# Créer un DataFrame avec les PassengerId et les prédictions
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})

# Sauvegarder le DataFrame au format CSV
submission.to_csv('my_submission.csv', index=False)