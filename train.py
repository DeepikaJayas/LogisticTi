import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("Titanic-Dataset.csv")

X = df[['Pclass', 'Age', 'Fare']].copy()
y = df['Survived']

X['Age'] = X['Age'].fillna(X['Age'].median())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

joblib.dump(model, "titanic_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ Model & Scaler trained with 3 features")
