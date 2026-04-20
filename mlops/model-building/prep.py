%%writefile mlops/model-building/prep.py
# data manipulation: work with series and dataframes
import pandas as pd
# split data into train and test sets
import sklearn
from sklearn.model_selection import train_test_split
# for preprocessor pipeline
from sklearn.compose import ColumnTransformer
# for one hot encoding of categorical columns
# for scaling of numerical columns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# work with folders
import os
# hf space authentication to upload files
from huggingface_hub import login, HfApi
# for working with year column
from datetime import datetime
# for model serialization format
import joblib

# define constants for dataset and output paths
DATASET_PATH = "hf://datasets/JaiBhatia020373/mlops/SuperKart.csv"
superkart_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully")

# keep a deep copy of original dataset as transactional set
dataset_trans = superkart_dataset.copy()

# drop the id columns: Product_Id, Store_Id from dataset
dataset_trans.drop(columns=["Product_Id"], inplace=True)
dataset_trans.drop(columns=["Store_Id"], inplace=True)

# compute the age of store and add the 'Age_Of_Store' column
current_year = datetime.now().year
dataset_trans['Age_of_Store'] = current_year - dataset_trans['Store_Establishment_Year']

# drop the Store_Establishment_Year column
dataset_trans.drop(columns=["Store_Establishment_Year"], inplace=True)

print(dataset_trans.head())

# define target variable for regression task
target = "Product_Store_Sales_Total"

# save a list of all input variables in X
X = dataset_trans.drop(columns=[target])

# define feature groups
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

y = dataset_trans[target]

print("Numeric features: ", numeric_features)
print("Categorical features: ", categorical_features)

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# define Preprocessing pipeline using column transformers
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ]
)

# Save to joblib after preprocessing
joblib.dump(preprocessor, "mlops/model-building/preprocessor.joblib")

# fit and transform preprocessor on the training set
X_train_transformed = preprocessor.fit_transform(X_train)

# after preprocessor trained on train set then transform on the test set
X_test_transformed = preprocessor.transform(X_test)

# get feature names list
feature_names = preprocessor.get_feature_names_out()
X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

# Save data as CSV
X_train_df.to_csv("train_prepared.csv", index=False)
X_test_df.to_csv("test_prepared.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# target repository: <login_id> / <repo master folder>
repo_id = "JaiBhatia020373" + "/" + "mlops"

# to manage/store model files
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

api.create_repo(repo_id=repo_id, 
                repo_type=repo_type, 
                exist_ok=True        # avoids error if it already exists
)

# upload preprocessor joblib to hf space
api.upload_file(
    path_or_fileobj="mlops/model-building/preprocessor.joblib",   # local file path
    path_in_repo="preprocessor.joblib",      # path inside repo
    repo_id=repo_id,
    repo_type=repo_type)

prepared_data_files = ["train_prepared.csv", "test_prepared.csv", "y_train.csv", "y_test.csv"]

# to manage/store data files
repo_type = "dataset"

# upload prepared data files to hf space
for file_path in prepared_data_files:
  api.upload_file(
    path_or_fileobj=file_path,
    path_in_repo=file_path,
    repo_id=repo_id,
    repo_type=repo_type)

print("Prepared csv files uploaded to hf space - data card.")
print("Preprocessing complete. Preprocessor file uploaded to hf space - model card.")
