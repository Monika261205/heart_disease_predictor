print("HEART DISEASE PREDICTION PROJECT STARTED SUCCESSFULLY!!")
#IMPORT REQUIRED LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#DISPLAY SETTINGS
pd.set_option('display.max_columns', None)
#load dataset
df=pd.read_csv("synthetic_heart_disease_dataset.csv")
print(df.columns)
df.head()
#shape of dataset
print("Dataset Shape:", df.shape)
#column information
df.info()
#statistical summary
df.describe()
df.isnull().sum()
#fill numeric  missing value with median(if any)
df=df.fillna(df.median(numeric_only= True))
#confirm again
df.isnull().sum()
df_numeric=df.select_dtypes(include=['number'])
plt.figure(figsize=(12,8))
sns.heatmap(df_numeric.corr(),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
print(df.columns)
X=df.drop('Heart_Disease', axis=1)
y=df["Heart_Disease"]
X.head()
# ---------------------------------------
# STEP 7: TRAIN–TEST SPLIT & PREPROCESSING
# ---------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# X = features, y = target (already defined in Step 6)
# Example:
# X = df.drop("target", axis=1)
# y = df["target"]

# ---------------------------------------
# Train-Test Split
# ---------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2,      # 20% test data
    random_state=42,    # keep results consistent
    stratify=y          # preserves class balance
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# ---------------------------------------
# Preprocessing (Feature Scaling)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Identify categorical and numeric columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
numeric_cols = X_train.select_dtypes(exclude=['object']).columns

# Create column transformer for encoding
ct = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Fit on training data and transform both train and test
X_train_encoded = ct.fit_transform(X_train)
X_test_encoded = ct.transform(X_test)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)
import pandas as pd

encoded_feature_names = ct.get_feature_names_out()

X_train_scaled = pd.DataFrame(X_train_scaled, columns=encoded_feature_names)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=encoded_feature_names)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Training the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predicting on the test set
y_pred = model.predict(X_test_scaled)

# Evaluating performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
import joblib

joblib.dump(model, "heart_disease_model.pkl")
joblib.dump(ct, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
import matplotlib.pyplot as plt
import numpy as np

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance")
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), np.array(encoded_feature_names)[indices], rotation=90)
plt.tight_layout()
plt.show()
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label="ROC Curve (AUC = {:.3f})".format(roc_auc))
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC Curve)")
plt.legend()
plt.show()
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision vs Recall")
plt.legend()
plt.show()
import numpy as np

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = np.array(encoded_feature_names)

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), features[indices], rotation=90)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Define the model
rf = RandomForestClassifier(random_state=42)

# Hyperparameter search space
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Random search
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit search
random_search.fit(X_train_scaled, y_train)

# Best model
best_rf = random_search.best_estimator_

print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)
y_pred_tuned = best_rf.predict(X_test_scaled)

print("Test Accuracy after tuning:", accuracy_score(y_test, y_pred_tuned))
print("\nClassification Report:\n", classification_report(y_test, y_pred_tuned))
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=2000)

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}

grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

print("Best Logistic Regression Params:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

best_log_reg = grid_search.best_estimator_
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

y_prob = best_rf.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve - Tuned RandomForest")
plt.legend()
plt.show()
from sklearn.metrics import accuracy_score

# Logistic Regression
y_pred_lr = model.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, y_pred_lr)

# Tuned Random Forest
y_pred_rf = best_rf.predict(X_test_scaled)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\nModel Accuracies:")
print("Logistic Regression:", acc_lr)
print("Tuned Random Forest:", acc_rf)

# Choose best model
if acc_rf > acc_lr:
    final_model = best_rf
    print("\nSelected Model: Tuned RandomForest")
else:
    final_model = model
    print("\nSelected Model: Logistic Regression")
import joblib

joblib.dump(final_model, "final_heart_disease_model.pkl")
joblib.dump(ct, "encoder.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nAll files saved: final_heart_disease_model.pkl, encoder.pkl, scaler.pkl")
model = joblib.load("final_heart_disease_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Example prediction
sample = X_test.iloc[0:1]         # raw input
encoded = encoder.transform(sample)
scaled = scaler.transform(encoded)

prediction = model.predict(scaled)
print("Prediction:", prediction)
