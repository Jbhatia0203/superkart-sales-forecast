%%writefile mlops/model-building/train.py
# work with series and data frames for data manipulation
import pandas as pd
# for numerical and scientific computations
import numpy as np
# load the serialized joblib file as a python object
import joblib
# regression model metrics
from sklearn.metrics import mean_squared_error, r2_score
# XGboost ensemble model
import xgboost as xgb
# Randomized Search for best hyper-parameters combination
from sklearn.model_selection import RandomizedSearchCV
from huggingface_hub import hf_hub_download

# load the preprocessor joblib
preprocessor = joblib.load("mlops/model-building/preprocessor.joblib")

repo_id = "JaiBhatia020373/mlops"
repo_type = "dataset"

files = ["train_prepared.csv", "test_prepared.csv", "y_train.csv", "y_test.csv"]

# Download the prepared csv files locally from hf space data repo
for filename in files:
  csv_path = hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    repo_type=repo_type   # or "model" if it's in a model repo
)
  if filename == "train_prepared.csv":
    X_train_path = csv_path
  elif filename == "test_prepared.csv":
    X_test_path = csv_path
  elif filename == "y_train.csv":
    y_train_path = csv_path
  elif filename == "y_test.csv":
    y_test_path = csv_path

X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path)
y_test = pd.read_csv(y_test_path)

# define xgboost model instance and initialize using same random generator seed
xgb_model = xgb.XGBRegressor(random_state=42)

# define a random search params grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
                     
# tuning the model with Randome Search parameters
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=20,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

# train the xgboost model with random-search params on training set
# since csv files were already fit and transformed with preprocessor
# hence make_pipeline not needed to be run
random_search.fit(X_train, y_train)

# get the best estimator for xgboost with random search
best_sales_forecast_model = random_search.best_estimator_

# Predict on test set
y_pred = best_sales_forecast_model.predict(X_test)

# Evaluate model performance - MSE, RMSE and R-Squared metrics
mse = np.round(mean_squared_error(y_test, y_pred), 2)
rmse = np.round(np.sqrt(mse), 2)
r2 = np.round(r2_score(y_test, y_pred), 2)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error", rmse )
print("R-Squared:", r2)

# to manage/store model files
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

api.create_repo(repo_id=repo_id, 
                repo_type=repo_type, 
                exist_ok=True        # avoids error if it already exists
)

# create joblib file for best estimator model
joblib.dump(best_sales_forecast_model, "mlops/model-building/sales-forecast.joblib")

# upload joblib file for best estimator model to hf space
api.upload_file(
    path_or_fileobj="mlops/model-building/sales-forecast.joblib",   # local file path
    path_in_repo="sales-forecast.joblib",      # path inside repo
    repo_id=repo_id,
    repo_type=repo_type)
