from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

def build_minimal_pipeline (X_train, y_train):
    """takes a cleaned dataframe and returns a trained minimal pipeline"""
    pipeline = Pipeline([("model", LinearRegression())])
    pipeline.fit(X_train, y_train)
    return pipeline
