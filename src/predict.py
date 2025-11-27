import pandas as pd
import joblib
import glob
import os

# --------------------------------------------
# LOAD ALL MODELS FROM ./models/
# --------------------------------------------
model_dir = "./models/"
model_paths = glob.glob(os.path.join(model_dir, "*.pkl"))

models = {}
for path in model_paths:
    model_name = os.path.basename(path).replace(".pkl", "")
    models[model_name] = joblib.load(path)

print(f"Loaded models: {list(models.keys())}")

# --------------------------------------------
# CREATE 10 DUMMY EXAMPLES
# --------------------------------------------
dummy_examples = [
    {
        "living_area (m²)": 200,
        "number_of_bedrooms": 3,
        "number_facades": 2,
        "open_fire": 1,
        "postal_code": 8370,
        "region": "Flanders",
        "terrace": 0,
        "swimming_pool": 0,
        "state_of_building": "New",
        "terrace_area (m²)": 50,
        "type": "house",
        "furnished": 1,
        "equiped_kitchen": 0,
        "garden": 1,
        "province": "West-Flanders",
        "subtype": "Residence"
    },
    {
        "living_area (m²)": 90,
        "number_of_bedrooms": 1,
        "number_facades": 4,
        "open_fire": 0,
        "postal_code": 1000,
        "region": "Brussels",
        "terrace": 1,
        "swimming_pool": 0,
        "state_of_building": "Good",
        "terrace_area (m²)": 12,
        "type": "apartment",
        "furnished": 0,
        "equiped_kitchen": 1,
        "garden": 0,
        "province": "Brussels",
        "subtype": "Penthouse"
    },
    # --- add remaining 8 examples here with exact column names ---
]

dummy_df = pd.DataFrame(dummy_examples)

# --------------------------------------------
# PREDICT WITH EACH MODEL
# --------------------------------------------
pred_results = dummy_df.copy()

for model_name, model in models.items():
    print(f"Predicting with {model_name}...")

    # Get features expected by the model
    if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
        # Pipeline with preprocessor
        model_features = model.named_steps["preprocessor"].get_feature_names_out()
    else:
        # fallback: use dummy_df columns
        model_features = dummy_df.columns

    # Align dummy_df to model's training features
    X_dummy_aligned = dummy_df.reindex(columns=model_features, fill_value=0)

    # Make predictions
    pred_results[f"prediction_{model_name}"] = model.predict(X_dummy_aligned).round(0).astype(int)

# --------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------
pred_results.to_csv("./data/predictions_dummy.csv", index=False)
print("\nSaved predictions to ./data/predictions_dummy.csv")
print(pred_results)
