import pandas as pd

from joblib import dump

from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import normalize, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

df = pd.read_csv('thyroid_cancer_risk_data.csv')
df.drop(
    ['Ethnicity'],
    axis = 1,
    inplace = True
)

le = LabelEncoder()
df['Gender'] = le.fit_transform(df["Gender"])
df['Country'] = le.fit_transform(df["Country"])
df['Family_History'] = le.fit_transform(df["Family_History"])
df['Radiation_Exposure'] = le.fit_transform(df["Radiation_Exposure"])
df['Iodine_Deficiency'] = le.fit_transform(df["Iodine_Deficiency"])
df['Smoking'] = le.fit_transform(df["Smoking"])
df['Obesity'] = le.fit_transform(df["Obesity"])
df['Diabetes'] = le.fit_transform(df["Diabetes"])
df['Thyroid_Cancer_Risk'] = le.fit_transform(df["Thyroid_Cancer_Risk"])
df['Diagnosis'] = le.fit_transform(df["Diagnosis"])

Y = df['Diagnosis']
X = df.drop('Diagnosis', axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

model = DecisionTreeClassifier(random_state=42)

param_grid = {
    'max_depth' :  [1, 2, 3, 5, 10, 15, 20],
    'min_samples_split': [1, 2, 3, 5, 10, 15],
    'criterion' : ['gini', 'entropy']
}

grid_search = GridSearchCV(
    model, 
    param_grid, 
    cv=8, 
    scoring='accuracy'
)

grid_search.fit(X_train, Y_train)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, Y_test)
print("Test Accuracy:", test_accuracy)

Ypred = best_model.predict(X_test)

print(confusion_matrix(Y_test, Ypred, labels=[0, 1]))