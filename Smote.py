import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.model_selection import GridSearchCV
from genetic_selection import GeneticSelectionCV

# Define feature extraction functions
def calculate_vmax(v):
    return v

def calculate_thd(hc, pff):
    return np.sum(hc * pff)

def calculate_hr(rr, t):
    return 60 * rr / t

def calculate_zcr(sc, snc):
    return sc / (sc + snc)

def calculate_entropy(pxy):
    return -np.sum(pxy * np.log(pxy))

def calculate_energy(pxy):
    return np.sum(pxy ** 2)

def calculate_sd(rr):
    m = len(rr)
    k = np.mean(np.diff(rr))
    return np.sqrt(np.sum((rr - np.roll(rr, -1)) ** 2) / (m - 1) - k ** 2)

def calculate_k(rr):
    return np.mean(np.diff(rr))

def calculate_r(rr):
    return np.sqrt(np.mean(np.diff(rr) ** 2))

# Load datasets (Assuming the datasets are loaded as X and y)
# Combine data and perform feature extraction
X_extracted = pd.DataFrame()
X_extracted['vmax'] = calculate_vmax(X['v'])
X_extracted['thd'] = calculate_thd(X['hc'], X['pff'])
X_extracted['hr'] = calculate_hr(X['rr'], X['t'])
X_extracted['zcr'] = calculate_zcr(X['sc'], X['snc'])
X_extracted['entropy'] = calculate_entropy(X['pxy'])
X_extracted['energy'] = calculate_energy(X['pxy'])
X_extracted['sd'] = calculate_sd(X['rr'])

# Feature selection using Recursive Feature Elimination with Cross-Validation (RFECV)
model = RandomForestClassifier()  # Random Forest Classifier for feature selection
rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')
X_selected = rfecv.fit_transform(X_extracted, y)

# Oversampling using SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_selected, y)

# Undersampling using Cluster Centroids
cc = ClusterCentroids()
X_resampled, y_resampled = cc.fit_resample(X_resampled, y_resampled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build MLDCNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reshape data for Conv1D input
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# Train the model
model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
y_pred = model.predict_classes(X_test_reshaped)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# ROC Curve and AUC
y_pred_proba = model.predict_proba(X_test_reshaped)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()



