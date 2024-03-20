import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from genetic_algorithm import GeneticAlgorithm  # You need to implement this
from elephant_herd_optimization import ElephantHerdOptimization  # You need to implement this

# Load data (replace this with your dataset)
data = pd.read_csv("heart_disease_data.csv")

# Data preprocessing
X = data.drop(columns=["target"])  # Features
y = data["target"]  # Target variable

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection using Recursive Feature Elimination (RFEM)
rfecv = RFECV(estimator=RandomForestClassifier(), step=1, cv=5, scoring='accuracy')
X_train_selected = rfecv.fit_transform(X_train_scaled, y_train)
X_test_selected = rfecv.transform(X_test_scaled)

# Data imbalance handling using USCOM
uscom = ClusterCentroids()
X_train_balanced, y_train_balanced = uscom.fit_resample(X_train_selected, y_train)

# Model training using MLDCNN with AEHOM
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, solver='adam', random_state=1)
eho = ElephantHerdOptimization()
best_params = eho.optimize(mlp, X_train_balanced, y_train_balanced)  # You need to implement optimization method
mlp.set_params(**best_params)
mlp.fit(X_train_balanced, y_train_balanced)

# Model evaluation
y_pred = mlp.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Cross-validation scores
cv_scores = cross_val_score(mlp, X_train_selected, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))

# Hyperparameter tuning
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (200,)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01]
}
grid_search = GridSearchCV(mlp, param_grid, cv=3)
grid_search.fit(X_train_balanced, y_train_balanced)
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()
