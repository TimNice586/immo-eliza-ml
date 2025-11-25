import pandas as pd
import numpy as np

def clean_data(df):
    """ This function drops duplicates once more to be sure
        drops the column property_ID since it does not affect price and is an immovlan code
        converts our datatypes (to the right type)
        drops properties without prices
        """
    
    # Drop duplicates (safety)
    df = df.drop_duplicates()

    # Drop useless columns
    df = df.drop(columns=["property_ID"], errors="ignore")

    # Convert datatypes
    df = df.apply(lambda x: x.astype("Int64") if x.dtype == float and (x.dropna() % 1 == 0).all() else x)

    # Explicit type conversions
    dtype_map = {
        "locality_name": "string",
        "province": "string",
        "region": "string",
        "type": "category",
        "subtype": "category",
        "state_of_building": "category",
        "postal_code": "category"
        
    }
    for col, dtype in dtype_map.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)

    # Drop rows without price
    df = df.dropna(subset=["price (€)"])

    #convert nullable dtypes pd.NA (pandas nans) to np.nan (numpy nans)
    #this must be done because later on transformers can not handle pd.NA
    df = df.replace({pd.NA: np.nan})

    return df

#testing
# df = pd.read_csv("./data/filtered_final_cleaned_data.csv")
# df_clean_test = clean_data(df)
# print(df_clean_test.dtypes)
# print(df.dtypes)