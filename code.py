import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 01 : Load the Dataset
df = pd.read_csv("LogisticTelecom_data.csv")
print(df.shape)
print(df.head(10))

# Step 02 : Inspect and Clean the Data
df.info()
df.isnull().sum()

# Step 3 — Encode Categorical Columns
# We must convert text categories (like “Yes/No”) into numeric values.
for column in df.columns:
 if df[column].dtype == 'object':
    df[column] = LabelEncoder().fit_transform(df[column])

# Step 04 : Split Data into Features and Target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Step 05 : Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 06 : Standardise the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7 — Train the Logistic Regression Model
model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)

# Step 08 : Make Predictions
y_pred = model.predict(X_test_scaled)

# Step 09 : Evaluate Model Performance
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print("Accuracy:", acc)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", cr)

# Step 10 : Visualise the Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
