print("Program Started")

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_csv("customer_churn.csv")

print("Dataset Loaded Successfully")
print(df.shape)

# Load dataset
df = pd.read_csv("customer_churn.csv")

# Encode categorical columns
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Target column
X = df.drop("Churn", axis=1)
y = df["Churn"]
print(X.columns.tolist())

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Accuracy
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))