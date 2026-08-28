import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

df['Amount_log'] = np.log1p(df['Amount'])
df = df.drop('Amount', axis=1)

scaler = StandardScaler()
df['Amount_log'] = scaler.fit_transform(df[['Amount_log']])
df['Time'] = scaler.fit_transform(df[['Time']])

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

contagem_classes = y_train.value_counts()
peso_fraudes = contagem_classes[0] / contagem_classes[1]

xgb_model = xgb.XGBClassifier(scale_pos_weight=peso_fraudes, random_state=42, eval_metric='auc', n_estimators=100)
xgb_model.fit(X_train_resampled, y_train_resampled)

y_proba = xgb_model.predict_proba(X_test)[:, 1]
y_pred_custom = (y_proba >= 0.3).astype(int)

print(classification_report(y_test, y_pred_custom))
print(f"AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

X_test_amostra = X_test.sample(n=1000, random_state=42)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test_amostra)
shap.summary_plot(shap_values, X_test_amostra, plot_type="dot")
plt.show()