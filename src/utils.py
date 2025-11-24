def clean_and_load():

    df = pd.read_csv("filtered_final_cleaned_data.csv")

    # converting float to int
    df = df.apply(lambda x: x.astype("Int64") if x.dtype == float and (x.dropna() % 1 == 0).all() else x)

    # converting objects to strings
    df['property_ID'] = df['property_ID'].astype('string')
    df['locality_name'] = df['locality_name'].astype('string')
    df['type'] = df['type'].astype('category')
    df['subtype'] = df['subtype'].astype('category')
    df['state_of_building'] = df['state_of_building'].astype('category')
    df['postal_code'] = df['postal_code'].astype('category')
    
    # removing properties that do not have the price
    df = df.dropna(subset=["price (€)"])
    
    return df

