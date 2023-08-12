from fastapi import FastAPI
from matplotlib import pyplot as plt
import uvicorn
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns 

working_directory = os.getcwd()
print(working_directory)
path = 'medical.xlsx'
df = pd.read_excel(path)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# df['gender'] = le.fit_transform(df['gender'])
df['smoking_status'] = le.fit_transform(df['smoking_status'])
df['htn'] = le.fit_transform(df['htn'])
df['dm'] = le.fit_transform(df['dm'])
df['ckd'] = le.fit_transform(df['ckd'])
df['rbbb'] = le.fit_transform(df['rbbb'])
df['lbbb'] = le.fit_transform(df['lbbb'])
df['baseline_qrs_interval'] = le.fit_transform(df['baseline_qrs_interval'])
df['new_onset_rbbb'] = le.fit_transform(df['new_onset_rbbb'])
df['new_onset_lbbb'] = le.fit_transform(df['new_onset_lbbb'])
df['pacemaker'] = le.fit_transform(df['pacemaker'])
df['asa'] = le.fit_transform(df['asa'])
df['AntiPlatelet_otherthanASA'] = le.fit_transform(df['AntiPlatelet_otherthanASA'])
df['anticoagulants'] = le.fit_transform(df['anticoagulants'])
df['beta_blocker'] = le.fit_transform(df['beta_blocker'])
# df['ccb'] = le.fit_transform(df['ccb'])
# df['acei_arb'] = le.fit_transform(df['acei_arb'])
# df['diuretics'] = le.fit_transform(df['diuretics'])
df['asa'] = le.fit_transform(df['asa'])
# df['aldosterone_antagonist'] = le.fit_transform(df['aldosterone_antagonist'])
# df['av_block'] = le.fit_transform(df['av_Block'])

df['hba1c'] = df['hba1c'].replace("ND", np.nan)
df['hba1c'] = pd.to_numeric(df['hba1c'])
df['hba1c'] = df['hba1c'].fillna(df['hba1c'].mean())
df['nt_proBnp'] = df['nt_proBnp'].replace("ND", np.nan)
df['nt_proBnp'] = pd.to_numeric(df['nt_proBnp'])
df['nt_proBnp'] = df['nt_proBnp'].fillna(df['nt_proBnp'].mean())
df['delta_pr_interval'] = df['delta_pr_interval'].replace("UK", np.nan)
df['delta_pr_interval'] = pd.to_numeric(df['delta_pr_interval'])
df['delta_pr_interval'] = df['delta_pr_interval'].fillna(df['delta_pr_interval'].mean())
df['asa'] = df['asa'].replace("NA", np.nan)
df['asa'] = pd.to_numeric(df['asa'])
df['asa'] = df['asa'].fillna(df['asa'].mean())
df['anticoagulants'] = df['anticoagulants'].replace("NA", np.nan)
df['anticoagulants'] = pd.to_numeric(df['anticoagulants'])
df['anticoagulants'] = df['anticoagulants'].fillna(df['anticoagulants'].mean())
df['beta_blocker'] = df['beta_blocker'].replace("NA", np.nan)
df['beta_blocker'] = pd.to_numeric(df['beta_blocker'])
df['beta_blocker'] = df['beta_blocker'].fillna(df['beta_blocker'].mean())
df['ccb'] = df['ccb'].replace("NA", np.nan)
df['ccb'] = pd.to_numeric(df['ccb'])
df['ccb'] = df['ccb'].fillna(df['ccb'].mean())
df['acei_arb'] = df['acei_arb'].replace("NA", np.nan)
df['acei_arb'] = pd.to_numeric(df['acei_arb'])
df['acei_arb'] = df['acei_arb'].fillna(df['acei_arb'].mean())
df['diuretics'] = df['diuretics'].replace("NA", np.nan)
df['diuretics'] = pd.to_numeric(df['diuretics'])
df['diuretics'] = df['diuretics'].fillna(df['diuretics'].mean())
df['aldosterone_antagonist'] = df['aldosterone_antagonist'].replace("NA", np.nan)
df['aldosterone_antagonist'] = pd.to_numeric(df['aldosterone_antagonist'])
df['aldosterone_antagonist'] = df['aldosterone_antagonist'].fillna(df['aldosterone_antagonist'].mean())
df['lvot_perimeter '] = df['lvot_perimeter '].fillna(df['lvot_perimeter '].mean())
df['lvot_diameter'] = df['lvot_diameter'].fillna(df['lvot_diameter'].mean())
df['valves_pressure'] = df['valves_pressure'].fillna(df['valves_pressure'].mean())
df['left_ventricle_size'] = df['left_ventricle_size'].fillna(df['left_ventricle_size'].mean())

df['birth'] = df['birth'].dt.year
df['date_of_tavr'] = df['date_of_tavr'].dt.month
# Add more features extraction as needed

# Apply numeric transformations
df['birth'] = df['birth'].fillna(df['birth'].mean())
df['date_of_tavr'] = df['date_of_tavr'].fillna(df['date_of_tavr'].mean())
print(df.isnull().sum())

x = df.drop("new_onset_lbbb", axis=1)
Y = df['new_onset_lbbb']


y = Y.to_numpy().reshape(-1)

# Apply label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
df.head(10)
k = 10
selector = SelectKBest(f_classif, k=k)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test_selected)

logistic_regression = LogisticRegression(random_state=42)
logistic_regression.fit(X_train_scaled, y_train_resampled)
y_pred_lr = logistic_regression.predict(X_test_scaled)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

knn = KNeighborsClassifier()
knn.fit(X_train_scaled, y_train_resampled)
y_pred_knn = knn.predict(X_test_scaled)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

svm = SVC(random_state=42)
svm.fit(X_train_scaled, y_train_resampled)
y_pred_svm = svm.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

gaussian_nb = GaussianNB()
gaussian_nb.fit(X_train_scaled, y_train_resampled)
y_pred_nb = gaussian_nb.predict(X_test_scaled)
print("GaussianNB Accuracy:", accuracy_score(y_test, y_pred_nb))

sgd = SGDClassifier(random_state=42)
sgd.fit(X_train_scaled, y_train_resampled)
y_pred_sgd = sgd.predict(X_test_scaled)
print("SGD Accuracy:", accuracy_score(y_test, y_pred_sgd))

xgb = XGBClassifier(random_state=42)
xgb.fit(X_train_scaled, y_train_resampled)
y_pred_xgb = xgb.predict(X_test_scaled)
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train_scaled, y_train_resampled)
y_pred_dt = decision_tree.predict(X_test_scaled)
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))

random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train_scaled, y_train_resampled)
y_pred_rf = random_forest.predict(X_test_scaled)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

voting_classifier = VotingClassifier(
    estimators=[
        ('lr', logistic_regression),
        ('knn', knn),
        ('svm', svm),
        ('nb', gaussian_nb),
        ('sgd', sgd),
        ('xgb', xgb),
        ('dt', decision_tree),
        ('rf', random_forest)
    ],
    voting='hard'
)
voting_classifier.fit(X_train_scaled, y_train_resampled)
y_pred_combined = voting_classifier.predict(X_test_scaled)
print("Voting Classifier Accuracy:", accuracy_score(y_test, y_pred_combined))

app = FastAPI()


class TAVR(BaseModel):
    Age: float
    Sex: int
    BSA: float
    BMI: float
    HTN: int
    CAD: int
    DM: int
    ACEi_ARB: int
    Beta_Blocker: int
    Aldosteroneantagonist: int
    CCB: int
    AntiPlateletotherthanASA: int
    ASA: int
    AntiplateletTherapy: int
    Diuretics: int
    LVEF: float
    SystolicBP: float
    DiastolicBP: float 
    LVOT: float
    ValveCode: int
    ValveSize: int
    BaselineRhythm: int
    PR: float
    QRS: int
    QRSmorethan120: int
    FirstdegreeAVblock: float
    Baseline_conduction_disorder: int
    BaselineRBBB: int
    DeltaPR: float
    DeltaQRS: int
    PacemakerImplantation: int

def main():
    return 'New-onset LBBB'

@app.get('/predict')
def hello():
    return {'reply from I have no idea now server'}


@app.post('/predict')
def predict(request: TAVR):
    input_data = request.dict()
    selected_features = selector.transform(pd.DataFrame([input_data]))
    scaled_input = scaler.transform(selected_features)
    prediction = int(voting_classifier.predict(scaled_input)[0])
    return {"prediction LBBB": prediction}

if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
    