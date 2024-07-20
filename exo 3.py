
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset
dataset = pd.read_csv("heart.csv")

# Afficher les premières lignes du dataset
print(dataset.head())

# Afficher les dimensions du dataset
print(dataset.shape)

# Afficher le type de données des colonnes du dataset
print(dataset.dtypes)

# Afficher les noms des colonnes
print(dataset.columns)

# Ajuster le nom de la colonne cible
target_column = 'output'  # Utiliser 'output' comme colonne cible

# Vérifier si la colonne cible existe dans le dataset
if target_column not in dataset.columns:
    raise ValueError(f"Colonne '{target_column}' non trouvée dans le dataset. Les colonnes disponibles sont: {dataset.columns}")

# Extraire les features : X
X = dataset.drop(columns=[target_column])

# Extraire la target : y
y = dataset[target_column]

# Normaliser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split le dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=23)
print(X_train.shape, X_test.shape)

# Créer le modèle
model = LogisticRegression(max_iter=200)  # Augmenter le nombre d'itérations

# Entrainer le modèle
model.fit(X_train, y_train)

# Afficher les paramètres du modèle
print(model.coef_, model.intercept_)

# Fonction de prédiction
def predict2(X):
    return model.predict(X)

# Prédictions
predictions = predict2(X_test)
print(predictions)

# Evaluation du modèle
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix: \n{cm}")

# Visualiser la matrice de confusion
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visualiser les scores
scores = {
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
}

plt.figure(figsize=(10, 5))
plt.bar(scores.keys(), scores.values(), color=['blue', 'green', 'red', 'purple'])
plt.ylim(0, 1)
plt.title('Model Evaluation Metrics')
plt.ylabel('Score')
plt.show()