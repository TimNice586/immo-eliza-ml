import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utils import clean_data
from pipeline import build_minimal_pipeline
import joblib

# Load data
df = pd.read_csv("./data/filtered_final_cleaned_data.csv")
df_clean = clean_data(df)

# Drop NaNs in selected features
selected_features = ["living_area (m²)", "number_of_bedrooms", "number_facades"]
df_clean = df_clean.dropna(subset=selected_features + ["price (€)"])

# Define X and y
X = df_clean[selected_features]
y = df_clean["price (€)"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train minimal pipeline
pipeline_minimal = build_minimal_pipeline(X_train, y_train)

# Evaluate
y_train_pred = pipeline_minimal.predict(X_train)
y_test_pred = pipeline_minimal.predict(X_test)

R2_train = r2_score(y_train, y_train_pred)
MAE_train = mean_absolute_error(y_train, y_train_pred)
MSE_train = mean_squared_error(y_train, y_train_pred)

R2_test = r2_score(y_test, y_test_pred)
MAE_test = mean_absolute_error(y_test, y_test_pred)
MSE_test = mean_squared_error(y_test, y_test_pred)


print("Training Metrics:")
print("R² test:", R2_train)
print("MAE test:", MAE_train)
print("MSE test:", MSE_train)

print("Testing Metrics")
print("R² test:", R2_test)
print("MAE test:", MAE_test)
print("MSE test:", MSE_test)
print("")

if R2_train >= R2_test:
    print("model is overfitting")
elif R2_train and R2_test < 0.5:
    print("model is weak and underfitting")
else:
    print("model is a good fit")

# Save model
joblib.dump(pipeline_minimal, "./models/minimal_linear_3features.pkl")