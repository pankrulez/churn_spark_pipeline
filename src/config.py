# src/config.py

# File Paths
INPUT_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_MODEL_PATH = "models/churn_logistic_regression"
OUTPUT_PREDICTIONS = "output/churn_predictions"

# App Name
APP_NAME = "TelcoChurnPrediction_Expert"

# Schema Constants (Good practice to avoid typo errors)
LABEL_COL = "Churn"
ID_COL = "customerID"
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", 
    "PhoneService", "MultipleLines", "InternetService", 
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
    "TechSupport", "StreamingTV", "StreamingMovies", 
    "Contract", "PaperlessBilling", "PaymentMethod"
]

# output model path
OUTPUT_MODEL_PATH = "models/churn_logistic_regression"

# output predictions path
OUTPUT_PREDICTIONS = "output/churn_predictions"