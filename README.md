# 🏡 Immo Eliza — Property Price Prediction (Machine Learning)

## 📊 Project Overview

This machine learning project focuses on predicting Belgian property prices using the Immo Eliza dataset.  
It includes a complete end-to-end pipeline:

- Data preprocessing  
- Feature engineering  
- Model training with hyperparameter tuning  
- Model evaluation  
- Model exporting  
- A ready-to-use prediction script (`predict.py`)

The goal is to build reliable, reproducible models capable of estimating realistic property prices based on property characteristics.

---

## 🎯 Key Objectives

- **Data Understanding** – Explore and analyze Belgian property data  
- **Preprocessing** – Clean, encode, and prepare the dataset  
- **Modeling** – Train multiple ML models  
- **Hyperparameter Optimization** – Improve performance using CV  
- **Model Saving** – Export pipelines for future use  
- **Prediction Pipeline** – Provide a user-friendly prediction script  

---

## 🤖 Models Implemented

The following machine learning models were developed and evaluated:

- **Linear Regression (LR)**
- **Random Forest Regressor (RF)**
- **XGBoost Regressor (XGB)**
- **Support Vector Regressor (SVM)**

All models use **scikit-learn Pipelines**, ensuring identical preprocessing during training and prediction
---

## 🔍 Technical Highlights

### 🔧 Data Processing
- Cleaning missing data  
- One-hot encoding for categorical features  
- Scaling features where necessary  
- Correct train/test/validation segregation to avoid leakage  

### 📈 Model Training & Tuning
- **RandomizedSearchCV** & **GridSearchCV**  
- Cross-validation  
- Separate early stopping procedure for XGBoost  
- Evaluation metrics:
  - R² (train/test)
  - Error analysis

### 📦 Model Export
All models are saved as `.pkl` files in the `models/` directory.

models/

├── LR_model.pkl

├── RF_model.pkl

├── XGB_model.pkl

└── SVM_model.pkl

---

## 🚀 Prediction Script (`predict.py`)

The project includes `predict.py`, which demonstrates how to load and run predictions with all trained models.

### Features:
- Loads every model pipeline  
- Creates **10 dummy example properties**  
- Predicts prices using all models  
- Saves prediction results into `data/predictions_dummy.csv`

Run it with:

```bash
python predict.py
This script serves as a practical template for integrating the trained models into real-world applications.

📁 Repository Structure
css
Copy code
├── data
│   ├── raw_data
│   ├── cleaned_data
│   └── predictions_dummy.csv
│
├── models
│   ├── LR_model.pkl
│   ├── RF_model.pkl
│   ├── XGB_model.pkl
│   └── SVM_model.pkl
│
├── notebooks
│   └── Analysis_Notebook.ipynb
│
├── src
│   ├── train.py
│   └── predict.py
│
├── README.md
└── requirements.txt
🗓 Timeline
This project was completed in four days, covering data analysis, modeling, evaluation, and deployment preparation.

👤 Author
Tim De Nijs
Data Science & AI — BeCode Ghent (2025–2026)