import pandas as pd
import joblib

#load the saved pipeline
model_path = "./models/full_LR_version_3b.pkl"
pipeline = joblib.load(model_path)

#load dummy examples
dummy_data = pd.DataFrame([{"living_area (m²)": 100, "number_of_bedrooms" : 3, "number_facades" : 2, "open_fire (yes:1, no:0)" : 1, "postal_code" : 8370, "region" : "Flanders" , 'terrace (yes:1, no:0)' : 0, 'swimming_pool (yes:1, no:0)' : 0, 'state_of_building' : "New", 'terrace_area (m²)' : 50, 'type' : "house", 'furnished (yes:1, no:0)' : 1, 'equiped_kitchen (yes:1, no:0)' : 0, 'garden (yes:1, no:0)': 1, 'province' : "West-Flanders", 'subtype': "Residence"}])

#we can also load from new csvs like so: dummy_data = pd.read_csv("./data/new_properties.csv")

#make predictions
predictions = pipeline.predict(dummy_data)

#predictiontime
dummy_data["price_prediction in €"] = predictions.round(0).astype(int)
print(dummy_data)

#we can save these predictions if we wish like so: dummy_data.to_csv("./data/predcitions_dummy.csv", index = False)