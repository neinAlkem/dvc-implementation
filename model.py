# Import Library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import joblib
import json

# Data Load and Quick Checks
data = pd.read_csv('dataset/raw/heart_1.csv')
data.head()
data.isnull().sum()

# Data Split and Scaling
y = data['target']
X = data.drop('target', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scale = StandardScaler()
X_train = scale.fit_transform(X_train)  # Fit and transform on training data
X_test = scale.transform(X_test)        # Only transform on testing data

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate Model 
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')  
recall = recall_score(y_test, y_pred, average='weighted')  
precision = precision_score(y_test, y_pred, average='weighted')  
report = classification_report(y_test, y_pred, output_dict=True)

# Simpan hasil evaluasi
evaluation_metrics = {
    "accuracy": accuracy,
    "f1_score": f1,
    "recall": recall,
    "precision": precision,
    "classification_report": report
}
with open('metrics.json', 'w') as f:
    json.dump(evaluation_metrics, f)

# Simpan Model
joblib.dump(model, 'model.pkl')
