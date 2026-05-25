# -------------------------------
# Updated production-ready XGBoost pipeline
# -------------------------------
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import joblib
import shap
import mlflow
import mlflow.xgboost
from pathlib import Path

# -------------------------------
# Load dataset
# -------------------------------
# Use local dataset in the same directory for production deployment
dataset_path = Path(__file__).resolve().parent / "gym_dataset.csv"
df = pd.read_csv(dataset_path)

# -------------------------------
# Handle missing values (mean imputation for numeric features)
# -------------------------------
for col in [
    'sleep_hours',
    'training_frequency_per_week',
    'calorie_surplus',
    'progressive_overload_score',
    'training_experience_years',
    'body_fat_percentage',
    'protein_intake_g',
    'stress_level'
]:
    df[col] = df[col].fillna(df[col].mean())

# -------------------------------
# Outlier handling (clip values to realistic ranges)
# -------------------------------
df["protein_intake_g"] = df["protein_intake_g"].clip(0, 300)
df["sleep_hours"] = df["sleep_hours"].clip(0, 12)
df["training_frequency_per_week"] = df["training_frequency_per_week"].clip(0, 14)
df["calorie_surplus"] = df["calorie_surplus"].clip(0, 2000)
df["progressive_overload_score"] = df["progressive_overload_score"].clip(0, 10)
df["training_experience_years"] = df["training_experience_years"].clip(0, 40)
df["body_fat_percentage"] = df["body_fat_percentage"].clip(3, 50)
df["stress_level"] = df["stress_level"].clip(0, 10)

# -------------------------------
# Separate features and target
# -------------------------------
X = df.drop("muscle_growth_likely", axis=1)
y = df["muscle_growth_likely"]

# Save feature order for deployment
feature_names = list(X.columns)
with open("feature_names.json", "w") as f:
    json.dump(feature_names, f)

# -------------------------------
# Train Validation Test split 70 15 15
# -------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# -------------------------------
# Handle imbalance using weighting
# -------------------------------
count_negative = np.sum(y_train == 0)
count_positive = np.sum(y_train == 1)
scale_pos_weight = count_negative / max(1, count_positive)
print("scale_pos_weight:", scale_pos_weight)

# -------------------------------
# Hyperparameter grid for tuning
# -------------------------------
param_grid = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [100, 200, 300],
    "subsample": [0.8, 1],
    "colsample_bytree": [0.8, 1],
    "scale_pos_weight": [scale_pos_weight]
}

# -------------------------------
# GridSearchCV setup
# -------------------------------
xgb_clf = xgb.XGBClassifier(eval_metric="logloss", random_state=42, verbosity=0)
grid = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    cv=3,
    scoring="f1",
    verbose=1,
    n_jobs=-1
)
grid.fit(X_train, y_train)

print("Best Params:", grid.best_params_)

# -------------------------------
# Train final model with best params and early stopping on validation set
# -------------------------------
best_model = grid.best_estimator_
best_model.fit(X_train, y_train)

# -------------------------------
# Predictions and probabilities on test set
# -------------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

# -------------------------------
# Evaluation metrics and plots
# -------------------------------
print("Classification Report")
print(classification_report(y_test, y_pred))
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
roc_auc = roc_auc_score(y_test, y_proba)
print("ROC-AUC:", roc_auc)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.title("ROC Curve")
plt.savefig("roc_curve.png")
plt.close()

# Precision-Recall Curve
prec, rec, _ = precision_recall_curve(y_test, y_proba)
plt.figure()
plt.plot(rec, prec, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.title("Precision-Recall Curve")
plt.savefig("pr_curve.png")
plt.close()

# -------------------------------
# Feature importance and SHAP explainability
# -------------------------------
plt.figure()
xgb.plot_importance(best_model, max_num_features=10)
plt.title("XGBoost Feature Importance")
plt.savefig("feature_importance.png")
plt.close()

explainer = shap.Explainer(best_model)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("shap_summary.png")
plt.close()
# -------------------------------
# Save model and artifacts
# -------------------------------
joblib.dump(best_model, "muscle_growth_model.pkl")
joblib.dump(feature_names, "feature_names.pkl")
print("model trained and saved successfully.")
print("feature names saved successfully.")
# -------------------------------
# Log experiment with MLflow
# -------------------------------
mlflow.xgboost.log_model(best_model, "xgboost-model")
mlflow.log_params(grid.best_params_)
mlflow.log_metric("roc_auc", roc_auc)


# print(df.isnull().sum())
# print(df.describe())
# print(df.info())






