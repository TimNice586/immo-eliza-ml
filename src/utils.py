
def clean_data(df):
    """ This function drops duplicates once more to be sure
        also drops the column property_ID since it does not affect price and is an immovlan code
        also converts our datatypes to the right type
        also drops properties without prices
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

    return df

