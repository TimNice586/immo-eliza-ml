import pandas as pd
import numpy as np

def clean_data(df, drop_high_card_cats=True):
    """
    Clean the dataframe:
    - Drop duplicates
    - Drop useless columns
    - Convert numeric-like floats to Int64 if all integers, else float
    - Convert categorical columns
    - Drop rows without target
    - Optionally drop high-cardinality categorical columns (like locality_name)
    - Convert any pd.NA to np.nan
    """

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop useless columns
    df = df.drop(columns=["property_ID"], errors="ignore")
    df = df.drop(columns=["locality_name"], errors="ignore")

    # Convert numeric columns
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    for col in numeric_cols:
        if (df[col].dropna() % 1 == 0).all():
            df[col] = df[col].astype("Int64")  # nullable integer
        else:
            df[col] = df[col].astype(float)

    # Explicit categorical/string conversions
    dtype_map = {
        "province": object,
        "region": object,
        "type": "category",
        "subtype": "category",
        "state_of_building": "category",
        "postal_code": "category"
    }
    for col, dtype in dtype_map.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    # Drop rows without target
    df = df.dropna(subset=["price (€)"])

    # Ensure pd.NA replaced by np.nan
    df = df.apply(lambda x: np.nan if x is pd.NA else x)

    return df
