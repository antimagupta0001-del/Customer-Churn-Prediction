# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# ====== EDIT path if needed ======
CSV_PATH = "churn_dataset.csv"   # <- make sure this file exists in the project folder

# 1. Load
df = pd.read_csv(CSV_PATH)

# 2. Quick cleaning (adapt if your CSV differs)
# Ensure TotalCharges numeric
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

# 3. Prepare features and target
target = 'Churn'
if target not in df.columns:
    raise ValueError("Target column 'Churn' not found in CSV.")

X = df.drop(columns=['customerID', target], errors='ignore')
y = (df[target] == 'Yes').astype(int)

# 4. Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
# if SeniorCitizen is int but used as numeric, keep it
categorical_cols = [c for c in X.columns if c not in numeric_cols]

# 5. Preprocessing
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# 6. Model pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestClassifier(n_estimators=200, random_state=42))
])

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 8. Fit
clf.fit(X_train, y_train)

# 9. Evaluate (simple)
from sklearn.metrics import classification_report, roc_auc_score
pred = clf.predict(X_test)
probs = clf.predict_proba(X_test)[:,1]
print(classification_report(y_test, pred))
print("ROC AUC:", roc_auc_score(y_test, probs))

# 10. Save pipeline
joblib.dump(clf, "churn_model_pipeline.pkl")
print("Saved pipeline to churn_model_pipeline.pkl")
