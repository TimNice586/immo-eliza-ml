import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from utils import clean_data
from pipeline import build_preprocessing_pipeline, build_full_pipeline, log_transform, exp_transform
import joblib
from scipy.stats import uniform, randint
import xgboost as xgb
from sklearn.pipeline import Pipeline

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

# keep copies of orig target for eval
y_train_orig = y_train.copy()
y_test_orig = y_test.copy()

#log-transform target training data if skewed
target_skewness = 1
if abs(y_train.skew()) > target_skewness:
    y_train = log_transform(y_train)
    target_log_transform = True
else:
    target_log_transform = False

#numeric features to log-transform if skewed
skew_treshold = 1
log_transform_features = [ numfeat for numfeat in numeric_features if abs(X_train[numfeat].skew()) > skew_treshold ]

#---------------
# Preprocessing
#---------------

#preprocess & build pipeline
preprocessor = build_preprocessing_pipeline(numeric_features, categorical_features, log_transform_features)


#----------------
# training models
#-----------------

models_to_train = ["LR", "RF","XGB", "SVM"]
results = {}

for model in models_to_train:
    print(f"\n Training {model} now, please wait...")

    pipeline = build_full_pipeline(preprocessor, model_type = model)
    
    if model == "LR":
        #LR : no hyperparametic tuning -> baseline is fine
        pipeline.fit(X_train, y_train)
        #predict it!
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        #transform log(y) to exp(log(y) = y back
        # transform log(y) predictions and y_train/y_test back if target was log-transformed
        if target_log_transform:
            y_train_pred = exp_transform(y_train_pred)
            y_test_pred = exp_transform(y_test_pred)
        
        #use orig targets for eval
        y_train_eval = y_train_orig
        y_test_eval = y_test_orig

        #metrics    
        R2_train = r2_score(y_train_eval, y_train_pred)
        MAE_train = mean_absolute_error(y_train_eval, y_train_pred)
        MSE_train = mean_squared_error(y_train_eval, y_train_pred)

        R2_test = r2_score(y_test_eval, y_test_pred)
        MAE_test = mean_absolute_error(y_test_eval, y_test_pred)
        MSE_test = mean_squared_error(y_test_eval, y_test_pred)


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
        joblib.dump(pipeline, f"./models/{model}_final_run.pkl")
    elif model == "RF":
        param_dist = {"model__n_estimators": [200, 300, 400], "model__max_depth": [8, 12, 16], "model__min_samples_split": [5, 10,15], "model__min_samples_leaf": [2, 4, 6], "model__max_features": ["sqrt", "log2"]}
        search = RandomizedSearchCV(pipeline, param_distributions = param_dist, n_iter = 20, scoring = "r2", cv = 3, verbose = 1, n_jobs = -1, random_state = 42)
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        print(f"Best parameters for {model}: {search.best_params_}")
        
        #predict it!
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        #transform log(y) to exp(log(y) = y back
        # transform log(y) predictions and y_train/y_test back if target was log-transformed
        if target_log_transform:
            y_train_pred = exp_transform(y_train_pred)
            y_test_pred = exp_transform(y_test_pred)
        
        #use orig targets for eval
        y_train_eval = y_train_orig
        y_test_eval = y_test_orig

        #metrics    
        R2_train = r2_score(y_train_eval, y_train_pred)
        MAE_train = mean_absolute_error(y_train_eval, y_train_pred)
        MSE_train = mean_squared_error(y_train_eval, y_train_pred)

        R2_test = r2_score(y_test_eval, y_test_pred)
        MAE_test = mean_absolute_error(y_test_eval, y_test_pred)
        MSE_test = mean_squared_error(y_test_eval, y_test_pred)


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
        joblib.dump(pipeline, f"./models/{model}_final_run.pkl")
    elif model == "XGB":
        # --------------------
        # Hyperparameter tuning with RandomizedSearchCV
        # --------------------
        X_train_base, X_val_base, y_train_base, y_val_base = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        xgb_for_search = xgb.XGBRegressor(objective="reg:squarederror", tree_method="hist", eval_metric="rmse", random_state=42, n_jobs=-1)

        search_pipeline = Pipeline([("preprocessor", preprocessor), ("model", xgb_for_search)])

        param_dist = {"model__n_estimators": [200, 400, 800],"model__max_depth": [4, 6, 8, 10],"model__learning_rate": [0.03, 0.05, 0.1],"model__subsample": [0.7, 0.9, 1.0],"model__colsample_bytree": [0.7, 0.9, 1.0],"model__reg_alpha": [0, 0.1, 0.5, 1],"model__reg_lambda": [0.5, 1, 2],"model__gamma": [0, 0.1, 0.2],}

        search = RandomizedSearchCV(search_pipeline,param_distributions=param_dist,n_iter=20,scoring="r2",cv=3,verbose=1,n_jobs=-1,random_state=42)
        search.fit(X_train_base, y_train_base)
        print("Best XGB params found:", search.best_params_)

        # --------------------
        # Extract best preprocessor and params
        # --------------------
        best_preprocessor = search.best_estimator_.named_steps["preprocessor"]
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}

        # Transform for xgboost
        X_train_pre = best_preprocessor.transform(X_train_base)
        X_val_pre = best_preprocessor.transform(X_val_base)
        dtrain = xgb.DMatrix(X_train_pre, label=y_train_base)
        dval = xgb.DMatrix(X_val_pre, label=y_val_base)

        xgb_params = {
            "objective": "reg:squarederror",
            "learning_rate": best_params["learning_rate"],
            "max_depth": best_params["max_depth"],
            "subsample": best_params["subsample"],
            "colsample_bytree": best_params["colsample_bytree"],
            "gamma": best_params["gamma"],
            "reg_alpha": best_params["reg_alpha"],
            "reg_lambda": best_params["reg_lambda"],
        }

        # --------------------
        # Train final XGB model with early stopping
        # --------------------
        print("\nRetraining best XGB with early stopping...")
        best_xgbmodel = xgb.train(
            params=xgb_params,
            dtrain=dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        print(f"XGB training finished. Best iteration: {best_xgbmodel.best_iteration}")
        #pipeline = Pipeline([("preprocessor", best_preprocessor),("model", best_xgbmodel)])
        
        #predict it!
        X_train_pre = best_preprocessor.transform(X_train)
        X_test_pre = best_preprocessor.transform(X_test)

        y_train_pred = best_xgbmodel.predict(xgb.DMatrix(X_train_pre))
        y_test_pred = best_xgbmodel.predict(xgb.DMatrix(X_test_pre))

        #transform log(y) to exp(log(y) = y back
        # transform log(y) predictions and y_train/y_test back if target was log-transformed
        if target_log_transform:
            y_train_pred = exp_transform(y_train_pred)
            y_test_pred = exp_transform(y_test_pred)
        
        #use orig targets for eval
        y_train_eval = y_train_orig
        y_test_eval = y_test_orig

        #metrics    
        R2_train = r2_score(y_train_eval, y_train_pred)
        MAE_train = mean_absolute_error(y_train_eval, y_train_pred)
        MSE_train = mean_squared_error(y_train_eval, y_train_pred)

        R2_test = r2_score(y_test_eval, y_test_pred)
        MAE_test = mean_absolute_error(y_test_eval, y_test_pred)
        MSE_test = mean_squared_error(y_test_eval, y_test_pred)


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
        joblib.dump(pipeline, f"./models/{model}_final_run.pkl")
    elif model == "SVM":
        param_dist = {"model__C": uniform(0.1, 10), "model__epsilon": uniform(0.01, 0.5), "model__gamma": ["scale", "auto"]}
        search = RandomizedSearchCV(pipeline, param_distributions = param_dist, n_iter = 20, scoring = "r2", cv = 3, verbose = 1, n_jobs = -1, random_state = 42)
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        print(f"Best parameters for {model}: {search.best_params_}")
        
        #predict it!
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        #transform log(y) to exp(log(y) = y back
        # transform log(y) predictions and y_train/y_test back if target was log-transformed
        if target_log_transform:
            y_train_pred = exp_transform(y_train_pred)
            y_test_pred = exp_transform(y_test_pred)
        
        #use orig targets for eval
        y_train_eval = y_train_orig
        y_test_eval = y_test_orig

        #metrics    
        R2_train = r2_score(y_train_eval, y_train_pred)
        MAE_train = mean_absolute_error(y_train_eval, y_train_pred)
        MSE_train = mean_squared_error(y_train_eval, y_train_pred)

        R2_test = r2_score(y_test_eval, y_test_pred)
        MAE_test = mean_absolute_error(y_test_eval, y_test_pred)
        MSE_test = mean_squared_error(y_test_eval, y_test_pred)


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
        joblib.dump(pipeline, f"./models/{model}_final_run.pkl")
    

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