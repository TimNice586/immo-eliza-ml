import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utils import clean_data
from pipeline import build_preprocessing_pipeline, build_full_pipeline, log_transform, exp_transform
import joblib

#--------------------
# Load data & clean
#-------------------

df = pd.read_csv("./data/filtered_final_cleaned_data.csv")
df_clean = clean_data(df)

#identify feature_types
numeric_features = df_clean.select_dtypes(include ="number").columns.tolist()
numeric_features.remove("price (€)") #target is not a feature!
categorical_features = df_clean.select_dtypes(include=["object","category","string"]).columns.tolist()



#split into features and target
X = df_clean.drop(columns=["price (€)"])
y = df_clean["price (€)"]

#split into train & test FIRST before PREPROC to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#log-transform target training data if skewed
target_skewness = 1
if abs(y_train.skew()) > target_skewness:
    y_train = log_transform(y_train)
    target_log_transform = True
else:
    target_log_transform = False

#numeric features to log-transform if skewed
skew_treshold = 1
log_transform_features = [ numfeat for numfeat in numeric_features if abs(df_clean[numfeat].skew()) > skew_treshold ]

#---------------
# Preprocessing
#---------------

#preprocess & build pipeline
preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features, log_transform_features)


#----------------
# training models
#-----------------

models_to_train = ["LR", "RF", "XGB", "SVM"]
results = {}

for model in models_to_train:
    print(f"\n Training {model} now, please wait...")

    pipeline = build_full_pipeline(preprocessor, model_type = model)
    pipeline.fit(X_train, y_train)

    #predict it!
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    #transform log(y) to exp(log(y) = y back
    if target_log_transform:
        y_train_pred = exp_transform(y_train_pred)
        y_test_pred = exp_transform(y_test_pred)

    #metrics    
    R2_train = r2_score(y_train, y_train_pred)
    MAE_train = mean_absolute_error(y_train, y_train_pred)
    MSE_train = mean_squared_error(y_train, y_train_pred)

    R2_test = r2_score(y_test, y_test_pred)
    MAE_test = mean_absolute_error(y_test, y_test_pred)
    MSE_test = mean_squared_error(y_test, y_test_pred)


    results[model] = {  "pipeline": pipeline,
                        "R2_train": R2_train, "MAE_train": MAE_train, "MSE_train": MSE_train,
                        "R2_test": R2_test, "MAE_test": MAE_test, "MSE_test": MSE_test}
    
    print(f"Training R²: {R2_train:.3f} | Testing R²: {R2_test:.3f}")

     # Feedback how good is our model performing
    if R2_train - R2_test > 0.1:
        print("Model might be overfitting....")
    elif R2_train < 0.5 and R2_test < 0.5:
        print("Model is weak / underfitting.")
    else:
        print("Model seems a good fit.")
    
    # Save model
    joblib.dump(pipeline, f"./models/{model}_model_first_run.pkl")

# ---------------------------
# Compare model metrics in a table
# ---------------------------
metrics_df = pd.DataFrame({k: {
    "R2_train": v["R2_train"],
    "R2_test": v["R2_test"],
    "MAE_test": v["MAE_test"],
    "MSE_test": v["MSE_test"]
} for k,v in results.items()}).T

print("\nModel comparison:")
print(metrics_df)

#=======================================
# PHASE 3A : minimal 3 features training
#=======================================

# # Drop NaNs in selected features
# selected_features = ["living_area (m²)", "number_of_bedrooms", "number_facades"]
# df_clean = df_clean.dropna(subset=selected_features + ["price (€)"])

# # Define X and y
# X = df_clean[selected_features]
# y = df_clean["price (€)"]

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train minimal pipeline
# pipeline_minimal = build_minimal_pipeline(X_train, y_train)

# # Evaluate
# y_train_pred = pipeline_minimal.predict(X_train)
# y_test_pred = pipeline_minimal.predict(X_test)

# R2_train = r2_score(y_train, y_train_pred)
# MAE_train = mean_absolute_error(y_train, y_train_pred)
# MSE_train = mean_squared_error(y_train, y_train_pred)

# R2_test = r2_score(y_test, y_test_pred)
# MAE_test = mean_absolute_error(y_test, y_test_pred)
# MSE_test = mean_squared_error(y_test, y_test_pred)


# print("Training Metrics:")
# print("R² test:", R2_train)
# print("MAE test:", MAE_train)
# print("MSE test:", MSE_train)

# print("Testing Metrics")
# print("R² test:", R2_test)
# print("MAE test:", MAE_test)
# print("MSE test:", MSE_test)
# print("")

# if R2_train >= R2_test:
#     print("model is overfitting")
# elif R2_train and R2_test < 0.5:
#     print("model is weak and underfitting")
# else:
#     print("model is a good fit")

# # Save model
# joblib.dump(pipeline_minimal, "./models/minimal_linear_3features.pkl")