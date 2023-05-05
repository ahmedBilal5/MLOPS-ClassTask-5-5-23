import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import mlflow

# Set MLflow experiment
mlflow.set_tracking_uri("http://localhost:6000")
mlflow.set_experiment("experiment 1")
# Load the Wine Quality dataset
df = pd.read_csv('preprocessed_data.csv', delimiter=';')
X = df.drop(columns=['quality'])
y = df['quality']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor()
    
with mlflow.start_run(run_name="my_run"):

    # Train XGBoost model
    params = {"objective": "reg:squarederror", "eval_metric": "rmse", "n_estimators": 100}
    model.fit(X_train, y_train)
    mlflow.log_params(params)
    mlflow.sklearn.log_model(model)

# Train a Gradient Boosting Regressor on the data

import joblib
joblib.dump(model, 'model.joblib')
