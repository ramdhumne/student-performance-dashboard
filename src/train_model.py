import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
df = pd.read_csv("../data/students.csv")

# Target
df["average"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3
y = df["average"]

# Features
X = df.drop(["math score","reading score","writing score","average"], axis=1)

# Categorical columns
cat_cols = X.columns

# Preprocessing
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Model pipeline
model = Pipeline([
    ("prep", preprocessor),
    ("reg", LinearRegression())
])

# Train
model.fit(X, y)

# Save model
joblib.dump(model, "../models/student_model.pkl")

print("Model trained and saved!")