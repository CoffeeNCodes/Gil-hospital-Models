
import os
from pstats import Stats
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from fastapi import FastAPI
from sklearn.metrics import confusion_matrix, accuracy_score
from pydantic import BaseModel
import uvicorn
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

working_directory = os.getcwd()
print(working_directory)
path = 'medical.xlsx'
data = pd.read_excel(path)

# Drop columns that are not relevant for modeling
data = data.drop(["birth", "date_of_tavr"], axis=1)

data['pacemaker'] = data['pacemaker'].apply(lambda x: 1 if x == 'Y' else 0)
data['gender'] = data['gender'].apply(lambda x: 1 if x == 'M' else 0)
data['smoking_status'] = data['smoking_status'].apply(lambda x: 1 if x == 'Y' else 0)
data['htn'] = data['htn'].apply(lambda x: 1 if x == 'Y' else 0)
data['dm'] = data['dm'].apply(lambda x: 1 if x == 'Y' else 0)
data['ckd'] = data['ckd'].apply(lambda x: 1 if x == 'Y' else 0)
data['rbbb'] = data['rbbb'].apply(lambda x: 1 if x == 'Y' else 0)
# data['lbbb'] = data['lbbb'].apply(lambda x: 1 if x == 'Y' else 0)
data['Av_block'] = data['Av_block'].apply(lambda x: 1 if x == 'Y' else 0)
data['baseline_qrs_interval'] = data['baseline_qrs_interval'].apply(lambda x: 1 if x == 'Y' else 0)
data['new_onset_rbbb'] = data['new_onset_rbbb'].apply(lambda x: 1 if x == 'Y' else 0)
data['new_onset_lbbb'] = data['new_onset_lbbb'].apply(lambda x: 1 if x == 'Y' else 0)

categorical_cols = ["smoking_status","htn","dm","ckd","rbbb","lbbb","Av_block","baseline_qrs_interval","new_onset_rbbb", "asa", "AntiPlatelet_otherthanASA", "anticoagulants",
                    "beta_blocker", "ccb", "acei_arb", "diuretics", "aldosterone_antagonist", "gender"]

numerical_cols = data.columns.difference(categorical_cols)
y = data["new_onset_lbbb"]

imputer_cat = SimpleImputer(strategy="most_frequent")
data[categorical_cols] = imputer_cat.fit_transform(data[categorical_cols])

for col in numerical_cols:

    data[col] = pd.to_numeric(data[col], errors="coerce")

# Impute missing values in numerical columns
imputer_num = SimpleImputer(strategy="mean")
X_numerical_imputed_values = imputer_num.fit_transform(data[numerical_cols])
X_numerical_imputed = pd.DataFrame(X_numerical_imputed_values, columns=numerical_cols)

# Perform one-hot encoding for categorical columns

encoder = OneHotEncoder(drop="first", handle_unknown='ignore')
encoded_cols = encoder.fit_transform(data[categorical_cols])
encoded_feature_names = encoder.get_feature_names_out(categorical_cols)

# Convert the sparse matrix to a dense array
encoded_cols_dense = encoded_cols.toarray()

# Create the DataFrame with proper column names
encoded_df = pd.DataFrame(encoded_cols_dense, columns=encoded_feature_names)

# Assuming that encoded_df and X_numerical_imputed have the same number of rows
combined_data = pd.concat([X_numerical_imputed, encoded_df], axis=1)
X_train, X_test, y_train, y_test = train_test_split(combined_data, y, test_size=0.25, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
y_test = y_test.astype(int)
print(y_train.value_counts())
print(y_test.value_counts())

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

prob_logistic = model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1
f1_logistic = f1_score(y_test, y_pred)  # Calculate F1-score
auc_logistic = roc_auc_score(y_test, prob_logistic)  # Calculate AUC


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# # Create a more complex neural network with regularization and early stopping
# nn_model = Sequential()
# nn_model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
# nn_model.add(Dropout(0.5))  # Adding dropout for regularization
# nn_model.add(Dense(32, activation='relu'))
# nn_model.add(Dropout(0.3))  # Adding dropout for regularization
# nn_model.add(Dense(1, activation='sigmoid'))

# # Compile the neural network with learning rate scheduling
# nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Implement early stopping and learning rate reduction
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)

# # Train the neural network
# nn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

# # ... (rest of the code remains the same)

# # Predict using the trained neural network
# nn_y_pred = nn_model.predict(X_test_scaled)
# nn_y_pred = (nn_y_pred > 0.5).astype(int)

# nn_conf_matrix = confusion_matrix(y_test, nn_y_pred)
# nn_accuracy = accuracy_score(y_test, nn_y_pred)

# # Print neural network results
# print("Neural Network Confusion Matrix:\n", nn_conf_matrix)
# print("Neural Network Accuracy:", nn_accuracy)

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
f1_rf = f1_score(y_test, y_pred_rf)  # Calculate F1-score
auc_rf = roc_auc_score(y_test, prob_rf)  # Calculate AUC

print("Random Forest Accuracy:", accuracy_rf)
print("Confusion Matrix Random:")
print(conf_matrix_rf)
print("F1 Score Random Forest:", f1_rf)
print("AUC Random Forest:", auc_rf)

# K-Nearest Neighbors
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

f1_knn = f1_score(y_test, y_pred_knn)  # Calculate F1-score
auc_knn = roc_auc_score(y_test, y_pred_knn)  # Calculate AUC

print("K-Nearest Neighbors Accuracy:", accuracy_knn)
print("F1 Score K-Nearest Neighbors:", f1_knn)
print("AUC knn:", auc_knn)

# SVC
svc_model = SVC(random_state=42)
svc_model.fit(X_train_scaled, y_train)
y_pred_svc = svc_model.predict(X_test_scaled)
accuracy_svc = accuracy_score(y_test, y_pred_svc)

prob_svc = svc_model.decision_function(X_test_scaled)  # Use decision function for probability estimation
f1_svc = f1_score(y_test, y_pred_svc)  # Calculate F1-score
auc_svc = roc_auc_score(y_test, prob_svc)  # Calculate AUC

print("SVC Accuracy:", accuracy_svc)
print("F1 Score SVC:", f1_svc)
print("AUC SVC:", auc_svc)

# Gaussian Naive Bayes
gnb_model = GaussianNB()
gnb_model.fit(X_train_scaled, y_train)
y_pred_gnb = gnb_model.predict(X_test_scaled)
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)

prob_gnb = gnb_model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1
f1_gnb = f1_score(y_test, y_pred_gnb)  # Calculate F1-score
auc_gnb = roc_auc_score(y_test, prob_gnb)  # Calculate AUC

print("Gaussian Naive Bayes Accuracy:", accuracy_gnb)
print("F1 Score Gaussian Naive Bayes:", f1_gnb)
print("AUC Gaussian Naive Bayes:", auc_gnb)

# SVM + SGD
sgd_svm_model = SGDClassifier(random_state=42)
sgd_svm_model.fit(X_train_scaled, y_train)
y_pred_sgd_svm = sgd_svm_model.predict(X_test_scaled)
accuracy_sgd_svm = accuracy_score(y_test, y_pred_sgd_svm)

prob_sgd_svm = sgd_svm_model.decision_function(X_test_scaled)  # Use decision function for probability estimation
f1_sgd_svm = f1_score(y_test, y_pred_sgd_svm)  # Calculate F1-score
auc_sgd_svm = roc_auc_score(y_test, prob_sgd_svm)  # Calculate AUC

print("SVM + SGD Accuracy:", accuracy_sgd_svm)
print("F1 Score SVM + SGD:", f1_sgd_svm)
print("AUC SVM + SGD:", auc_sgd_svm)

# Extreme Gradient Boosting (XGBoost)
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train_scaled, y_train)
y_pred_xgb = xgb_model.predict(X_test_scaled)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)

prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1
f1_xgb = f1_score(y_test, y_pred_xgb)  # Calculate F1-score
auc_xgb = roc_auc_score(y_test, prob_xgb)  # Calculate AUC

print("XGBoost Accuracy:", accuracy_xgb)
print("F1 Score XGBoost:", f1_xgb)
print("AUC XGBoost:", auc_xgb)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_test_scaled)
accuracy_dt = accuracy_score(y_test, y_pred_dt)

prob_dt = dt_model.predict_proba(X_test_scaled)[:, 1]  # Probability of class 1
f1_dt = f1_score(y_test, y_pred_dt)  # Calculate F1-score
auc_dt = roc_auc_score(y_test, prob_dt)  # Calculate AUC

print("Decision Tree Accuracy:", accuracy_dt)
print("F1 Score Decision Tree:", f1_dt)
print("AUC Decision Tree:", auc_dt)

import matplotlib.pyplot as plt
import seaborn as sns

# gender_pacemaker_df = pd.DataFrame({'Gender': data['gender'], 'PacemakerImplantationRisk': y})
# gender_pacemaker_counts = gender_pacemaker_df.groupby(['Gender', 'PacemakerImplantationRisk']).size().unstack()

# sns.barplot(data=gender_pacemaker_counts.reset_index(), x='Gender', y=1)  # 1 indicates risk of pacemaker implantation
# plt.xlabel('Gender')
# plt.ylabel('Count')
# plt.title('Pacemaker Implantation Risk by Gender')
# plt.show()


# Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('logistic', model),
    ('random_forest', random_forest_model),
    ('knn', knn_model),
    ('svc', svc_model),
    ('gnb', gnb_model),
    ('sgd_svm', sgd_svm_model),
    ('xgb', xgb_model),
    ('dt', dt_model)
], voting='hard')
voting_clf.fit(X_train_scaled, y_train)
y_pred_voting = voting_clf.predict(X_test_scaled)
accuracy_voting = accuracy_score(y_test, y_pred_voting)
print("Voting Classifier Accuracy:", accuracy_voting)
voting_conf_matrix = confusion_matrix(y_test, y_pred_voting)
print("Confusion Matrix - Voting Classifier:")
print(voting_conf_matrix)



app = FastAPI()

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
    pacemaker:str


class PredictionOutput(BaseModel):
    lbbb_prediction: int  # Assuming 0 or 1



@app.get('/')
def hello():
    return {'hello': 'server is working'}

@app.post('/')
def predict(data: PredictionInput):

    # Convert the input data to a DataFrame
    input_data = pd.DataFrame([data.dict()])

    # Perform the same preprocessing as done during training
    input_data['pacemaker'] = input_data['pacemaker'].apply(lambda x: 1 if x == 'Y' else 0)
    input_data['gender'] = input_data['gender'].apply(lambda x: 1 if x == 'M' else 0)
    input_data['smoking_status'] = input_data['smoking_status'].apply(lambda x: 1 if x == 'Y' else 0)
    input_data['htn'] = input_data['htn'].apply(lambda x: 1 if x == 'Y' else 0)
    input_data['dm'] = input_data['dm'].apply(lambda x: 1 if x == 'Y' else 0)
    input_data['ckd'] = input_data['ckd'].apply(lambda x: 1 if x == 'Y' else 0)
    input_data['rbbb'] = input_data['rbbb'].apply(lambda x: 1 if x == 'Y' else 0)
    input_data['Av_block'] = input_data['Av_block'].apply(lambda x: 1 if x == 'Y' else 0)
    input_data['baseline_qrs_interval'] = input_data['baseline_qrs_interval'].apply(lambda x: 1 if x == 'Y' else 0)
    input_data['new_onset_rbbb'] = input_data['new_onset_rbbb'].apply(lambda x: 1 if x == 'Y' else 0)
    input_data['new_onset_lbbb'] = input_data['new_onset_lbbb'].apply(lambda x: 1 if x == 'Y' else 0)

    input_data[categorical_cols] = imputer_cat.transform(input_data[categorical_cols])
    input_data[numerical_cols] = imputer_num.transform(input_data[numerical_cols])
    input_encoded_cols = encoder.transform(input_data[categorical_cols])
    input_encoded_cols_dense = input_encoded_cols.toarray()
    input_encoded_df = pd.DataFrame(input_encoded_cols_dense, columns=encoded_feature_names)
    input_combined_data = pd.concat([input_data[numerical_cols], input_encoded_df], axis=1)
    input_scaled = scaler.transform(input_combined_data)

    # Make predictions using the model
    prediction = voting_clf.predict(input_scaled)


    return PredictionOutput(lbbb_prediction=int(prediction[0]))
if __name__ == "__main__":
    uvicorn.run("model2:app", host="0.0.0.0", port=8002)




