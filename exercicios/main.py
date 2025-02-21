import pandas as pd

from joblib import dump

from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt

df = pd.read_csv('water_potability.csv')

df.dropna(axis = 1, inplace = True)

Y = df['Potability']
X = df.drop('Potability', axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

model = LinearSVC(random_state=42)

param_grid = {
    'max_depth' :  [ 5, 10, 20, 40],
    'min_samples_split': [2, 5, 10, 15],
    'criterion' : ['gini', 'entropy']
}

grid_search = GridSearchCV(
    model, 
    param_grid, 
    cv=8, 
    scoring='accuracy'
)

model.fit(X_train, Y_train)

# print("Best Hyperparameters:", grid_search.best_params_)
# print("Best Accuracy:", grid_search.best_score_)

# best_model = grid_search.best_estimator_
# test_accuracy = best_model.score(X_test, Y_test)
# print("Test Accuracy:", test_accuracy)

Ypred = model.predict(X_test)

print(confusion_matrix(Y_test, Ypred, labels=[0, 1]))
print(mean_absolute_error(Y_test, Ypred))
print(mean_absolute_error(Y_train, model.predict(X_train)))

dump(model, 'water.pkl')