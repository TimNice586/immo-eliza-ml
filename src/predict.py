import pandas as pd
import joblib

#load the saved minimal pipeline
model_path = "./models/minimal_linear_3features.pkl"
pipeline = joblib.load(model_path)

#load dummy examples
dummy_data = pd.DataFrame([{"living_area (m²)": 100, "number_of_bedrooms" : 3, "number_facades" : 2}, {"living_area (m²)": 2000, "number_of_bedrooms" : 20, "number_facades" : 4} ])

#we can also load from new csvs like so: dummy_data = pd.read_csv("./data/new_properties.csv")

#make predictions
predictions = pipeline.predict(dummy_data)

#predictiontime
dummy_data["price_prediction in €"] = predictions.round(0).astype(int)
print(dummy_data)

#we can save these predictions if we wish like so: dummy_data.to_csv("./data/predcitions_dummy.csv", index = False)