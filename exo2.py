import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Générer un dataset synthétique
X, y = make_regression(n_samples=100, n_features=2, noise=0.1)
dataset = pd.DataFrame(X, columns=['Feature1', 'Feature2'])
dataset['Target'] = y

print(dataset.shape)

print(dataset.dtypes)

print(dataset.shape)

print(dataset['Target'].dtype)

print(len(dataset.columns) - 1)

X = dataset[['Feature1', 'Feature2']]

y = dataset['Target']

print(X.dtypes)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

print(model.coef_, model.intercept_)

print(model.coef_, model.intercept_)

def predict1(X):
    return model.predict(X)

predictions = predict1(X_test)
print(predictions)


mse = mean_squared_error(y_test, predictions)
print(mse)
# Interprétation : La MSE mesure l'écart moyen quadratique entre les valeurs prédites et les valeurs réelles. Une valeur plus basse indique un meilleur modèle.


r2 = r2_score(y_test, predictions)
print(r2)
# Interprétation : Le R² indique la proportion de la variance des données expliquée par le modèle. Une valeur proche de 1 indique un modèle performant.

def evaluer(y_hat, y_test):
    mse = np.mean((y_hat - y_test) ** 2)
    r2 = 1 - (np.sum((y_test - y_hat) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
    return mse, r2

mse_manual, r2_manual = evaluer(predictions, y_test)
print(mse_manual, r2_manual)




train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean, label="Train")
plt.plot(train_sizes, test_mean, label="Test")
plt.xlabel("Training Examples")
plt.ylabel("Score")
plt.title("Learning Curve")
plt.legend()
plt.show()
