
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from fastapi import FastAPI
from sklearn.metrics import confusion_matrix, accuracy_score
from pydantic import BaseModel
import uvicorn

working_directory = os.getcwd()
print(working_directory)
path = 'medical.xlsx'
data = pd.read_excel(path)

# Drop columns that are not relevant for modeling
data = data.drop(["birth", "date_of_tavr"], axis=1)

# Convert gender to binary
data['pacemaker'] = data['pacemaker'].apply(lambda x: 1 if x == 'Y' else 0)
data['gender'] = data['gender'].apply(lambda x: 1 if x == 'M' else 0)
data['smoking_status'] = data['smoking_status'].apply(lambda x: 1 if x == 'Y' else 0)
data['htn'] = data['htn'].apply(lambda x: 1 if x == 'Y' else 0)
data['dm'] = data['dm'].apply(lambda x: 1 if x == 'Y' else 0)
data['ckd'] = data['ckd'].apply(lambda x: 1 if x == 'Y' else 0)
data['rbbb'] = data['rbbb'].apply(lambda x: 1 if x == 'Y' else 0)
data['lbbb'] = data['lbbb'].apply(lambda x: 1 if x == 'Y' else 0)
data['Av_block'] = data['Av_block'].apply(lambda x: 1 if x == 'Y' else 0)
data['baseline_qrs_interval'] = data['baseline_qrs_interval'].apply(lambda x: 1 if x == 'Y' else 0)
data['new_onset_rbbb'] = data['new_onset_rbbb'].apply(lambda x: 1 if x == 'Y' else 0)
data['new_onset_lbbb'] = data['new_onset_lbbb'].apply(lambda x: 1 if x == 'Y' else 0)
categorical_cols = ["smoking_status","htn","dm","ckd","rbbb","lbbb","Av_block","baseline_qrs_interval","new_onset_rbbb","new_onset_lbbb","pacemaker", "asa", "AntiPlatelet_otherthanASA", "anticoagulants",
                    "beta_blocker", "ccb", "acei_arb", "diuretics", "aldosterone_antagonist", "gender"]

numerical_cols = data.columns.difference(categorical_cols)
y = data["pacemaker"]

imputer_cat = SimpleImputer(strategy="most_frequent")
data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])

for col in numerical_cols:

    data[col] = pd.to_numeric(data[col], errors="coerce")

# Impute missing values in numerical columns
imputer_num = SimpleImputer(strategy="mean")
X_numerical_imputed_values = imputer_num.fit_transform(data[numerical_cols])
X_numerical_imputed = pd.DataFrame(X_numerical_imputed_values, columns=numerical_cols)

# Perform one-hot encoding for categorical columns
encoder = OneHotEncoder(drop="first")
encoded_cols = encoder.fit_transform(data[categorical_cols])
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)

# Convert the sparse matrix to a dense array
encoded_cols_dense = encoded_cols.toarray()

# Create the DataFrame with proper column names
encoded_df = pd.DataFrame(encoded_cols_dense, columns=encoded_feature_names)


# Assuming that encoded_df and X_numerical_imputed have the same number of rows
combined_data = pd.concat([X_numerical_imputed, encoded_df], axis=1)
X_train, X_test, y_train, y_test = train_test_split(combined_data, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

prob_logistic = model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1
f1_logistic = f1_score(y_test, y_pred)  # Calculate F1-score
auc_logistic = roc_auc_score(y_test, prob_logistic)  # Calculate AUC

print("Logistic Regression Accuracy:", accuracy)
print("Confusion Matrix Logistic:")
print(conf_matrix)
print("F1 Score Logistic:", f1_logistic)
print("AUC Logistic:", auc_logistic)

# Random Forest
random_forest_model = RandomForestClassifier(random_state=42)
random_forest_model.fit(X_train_scaled, y_train)
y_pred_rf = random_forest_model.predict(X_test_scaled)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

prob_rf = random_forest_model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1
f1_rf = f1_score(y_test, y_pred_rf) 
auc_rf = roc_auc_score(y_test, prob_rf) 

print("Random Forest Accuracy:", accuracy_rf)
print("Confusion Matrix Random:")
print(conf_matrix_rf)
print("F1 Score Random Forest:", f1_rf)
print("AUC Random Forest:", auc_rf)

class PredictionInput(BaseModel):
    age: int
    gender: str
    height: float
    bmi: float
    weight: float
    smoking_status: str
    htn: str
    dm: str
    ckd: str
    hemoglobin: float
    hba1c: float
    ast: int
    alt: int
    creatinine: float
    nt_proBnp: float
    sBP: int 
    dBP: int
    pulse_rate: int
    respiratory_rate: int
    lvot_perimeter: float
    lvot_diameter: float
    valves_pressure: int
    left_ventricle_size: int
    rbbb: str
    lbbb: str
    pr_interval: int
    Av_block: str
    qrs_interval: int
    baseline_qrs_interval: str
    delta_pr_interval: int
    delta_qrs_interval: int
    new_onset_rbbb: str
    new_onset_lbbb: str
    asa: str
    AntiPlatelet_otherthanASA: str
    anticoagulants: str
    beta_blocker: str
    ccb: str
    acei_arb: str
    diuretics: str
    aldosterone_antagonist: str


class PredictionOutput(BaseModel):
    Pacemaker_prediction: int  # Assuming 0 or 1

app = FastAPI()

@app.get('/')
def get_prediction():
    return {"message": "GET request successful 2"}

@app.post('/')
def predict(data: PredictionInput):
    prediction_logistic = model.predict(X_test_scaled)
    prediction_rf = random_forest_model.predict(X_test_scaled)

    return PredictionOutput(Pacemaker_prediction=prediction_logistic[0])

if __name__ == "__main__":
    uvicorn.run("model1:app", host="0.0.0.0", port=8001)
