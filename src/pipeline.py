import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR

#-------------
# transformers
#-------------

def log_transform(X):
    """log transformer for skewed numeric feature distribution"""
    return np.log1p(X)

def exp_transform(Y):
    """exp transformer to get original feature/target back"""
    return np.expm1(Y)


#-------------------------------------
# preprocessing pipeline (all features)
#-------------------------------------

def build_preprocessing_pipeline(numeric_features, categorical_features, log_transform_features = None):
    """build a first full preprocessing pipeline and improve later per feature if necessary
        numeric: impute median + scale
        categorical: impute most_frequent + OHE
        log_transform : log-transforms
        """
    
    if log_transform_features is None:
        log_transform_features = []

    # pipeline for numeric features
    transformers = []

    # num features that do not get log-transformed (if any)
    numeric_not_log = [numfeat for numfeat in numeric_features if numfeat not in log_transform_features]

    #add the not log features to transformers list
    if numeric_not_log:
        numeric_no_log_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
        transformers.append(("num", numeric_no_log_transformer, numeric_not_log))
    #add the log features to transformers list
    if log_transform_features:
        numeric_log_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("log", FunctionTransformer(func=log_transform, validate=False)), ("scaler", StandardScaler())])
        transformers.append(("num_log", numeric_log_transformer, log_transform_features))

    #categorical preprocessing
    categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    transformers.append(("cat", categorical_transformer, categorical_features))

    #combine preproc for num & cat features syntax: ColumnTransformer(transformers=[('name', transformer, columns)]
    preprocessor = ColumnTransformer(transformers=transformers, remainder = "drop")

    return preprocessor


#---------------------------------------
# full pipelines (attach any model here)
# lr : Linear Regression
# rf : Random Forest
# xgb : XGBoost
# svm : support vector machine
#---------------------------------------

def build_full_pipeline(preprocessor, model_type):
    """attach a regression model to the preproccesing pipeline manually
        lr: Linear Regression (baseline)
        later: RF, XGBOOST, SVM
        train.py does this now auto"""
    
    if model_type == "LR":
        model = LinearRegression()
    elif model_type == "RF":
        model = RandomForestRegressor(n_estimators = 200, max_depth = None, random_state = 42, n_jobs = 1)
    elif model_type == "XGB":
        model = XGBRegressor(n_estimators = 200, max_depth = None, learning_rate = 0.1, random_state = 42, n_jobs = -1, tree_method = "hist", objective = "reg:squarederror", eval_metric="rmse")
    elif model_type == "SVM":
        model = SVR(kernel = "rbf", C=100, gamma="scale", epsilon = 0.1 )
    else:
        raise ValueError("model type must be: 'LR', 'RF', 'XGB', or 'SVM'")

    pipeline = Pipeline([("preprocessing", preprocessor), ("model", model)])

    return pipeline

#---------------------------------------
# minimal pipeline (3 features baseline) (old code)
#----------------------------------------

def build_minimal_pipeline (X_train, y_train):
    """takes a cleaned dataframe and returns a trained minimal pipeline (BASELINE)"""
    pipeline = Pipeline([("model", LinearRegression())])
    pipeline.fit(X_train, y_train)
    return pipeline
