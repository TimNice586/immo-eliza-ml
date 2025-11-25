import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

def build_minimal_pipeline (X_train, y_train):
    """takes a cleaned dataframe and returns a trained minimal pipeline (BASELINE)"""
    pipeline = Pipeline([("model", LinearRegression())])
    pipeline.fit(X_train, y_train)
    return pipeline

def build_preprocessing_pipeline(numeric_features, categorical_features):
    """build a first full preprocessing pipeline and improve later per feature if necessary
        numeric: impute median + scale
        categorical: impute most_frequent + OHE"""
    
    #numeric preprocessing
    numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer (strategy = "median")), ("scaler", StandardScaler())])

    #categorical preprocessing
    categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignor", sparse_output=False))])

    #combine preproc for num & cat features
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features),("cat", categorical_transformer, categorical_features)])

    return preprocessor

def build_full_pipeline(preprocessor)
    """attach a regression model to the preproccesing pipeline
        for now : Linear Regression (baseline)
        later: RF, XGBOOST, SVM"""
    
    full_pipeline = Pipeline(steps=[("preprocessing", preprocessor), ("model", LinearRegression())])

    return full_pipeline
