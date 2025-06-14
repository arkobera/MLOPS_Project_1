import os
from datetime import date

# For MongoDB connection
DATABASE_NAME = "Proj1"
COLLECTION_NAME = "Proj1-Data"
MONGODB_URL_KEY = "MONGODB_URL"

PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME = "model.pkl"

TARGET_COLUMN = "Calories"
CURRENT_YEAR = date.today().year
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "Proj1-Data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "xgb.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 1
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")
MODEL_TRAINER_MAX_DEPTH: int = 8
MODEL_TRAINER_COLSAMPLE_BYTREE: float = 0.9
MODEL_TRAINER_SUBSAMPLE: float = 0.9
MODEL_TRAINER_N_ESTIMATORS: int = 1000
MODEL_TRAINER_LEARNING_RATE: float = 0.007
MODEL_TRAINER_RANDOM_STATE: int = 0
MODEL_TRAINER_EVAL_METRIC: str = "rmse"
MODEL_FOLDS: int = 5

"""
MODEL Evaluation related constants
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.001
MODEL_BUCKET_NAME = "my-model-mlopsproj-07"
MODEL_PUSHER_S3_KEY = "model-registry"


APP_HOST = "0.0.0.0"
APP_PORT = 5000