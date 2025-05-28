# Breast Cancer Classification - Complete Machine Learning Pipeline
# Includes EDA, Preprocessing, Model Training, and Evaluation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                            classification_report, roc_auc_score, roc_curve)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 1. Load and Explore Data
print("\n=== Loading and Exploring Data ===")
data = pd.read_csv('data.csv')
print(f"\nDataset shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())
print("\nData types:")
print(data.dtypes)
print("\nMissing values:")
print(data.isnull().sum())

# 2. Data Cleaning
print("\n=== Data Cleaning ===")
data = data.drop('id', axis=1)
print("\nAfter dropping 'id' column:")
print(f"New shape: {data.shape}")

# 3. Exploratory Data Analysis (EDA)
print("\n=== Exploratory Data Analysis ===")

# Target variable distribution
plt.figure(figsize=(6,4))
sns.countplot(x='diagnosis', data=data)
plt.title('Diagnosis Distribution (M=Malignant, B=Benign)')
plt.show()

print("\nTarget variable distribution:")
print(data['diagnosis'].value_counts())

# Correlation analysis
plt.figure(figsize=(12,8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()

# 4. Data Preprocessing
print("\n=== Data Preprocessing ===")

# Encode target variable
y = data['diagnosis'].map({'M':1, 'B':0})
X = data.drop('diagnosis', axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Feature selection
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

selected_features = X.columns[selector.get_support()]
print("\nSelected features:")
print(selected_features.tolist())

# 5. Model Training
print("\n=== Model Training ===")

models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_selected, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_selected)
    y_prob = model.predict_proba(X_test_selected)[:, 1]
    
    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    # Print metrics
    print(f"{name} Accuracy: {results[name]['accuracy']:.4f}")
    print(f"{name} ROC AUC: {results[name]['roc_auc']:.4f}")

# 6. Model Evaluation
print("\n=== Model Evaluation ===")

# ROC Curves
plt.figure(figsize=(8,6))
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['model'].predict_proba(X_test_selected)[:, 1])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.show()

# Feature Importance (Random Forest)
if 'Random Forest' in results:
    importances = results['Random Forest']['model'].feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10,6))
    plt.title('Random Forest Feature Importances')
    plt.bar(range(len(selected_features)), importances[indices])
    plt.xticks(range(len(selected_features)), selected_features[indices], rotation=90)
    plt.tight_layout()
    plt.show()

# 7. PCA Visualization
print("\n=== PCA Visualization ===")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[y_train == 0, 0], X_pca[y_train == 0, 1], color='blue', label='Benign', alpha=0.5)
plt.scatter(X_pca[y_train == 1, 0], X_pca[y_train == 1, 1], color='red', label='Malignant', alpha=0.5)
plt.title('PCA of Breast Cancer Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# 8. Final Results
print("\n=== Final Results ===")
best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\nBest model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.4f}")

print("\nClassification Report for Best Model:")
print(best_model[1]['classification_report'])

print("\nConfusion Matrix for Best Model:")
print(best_model[1]['confusion_matrix'])