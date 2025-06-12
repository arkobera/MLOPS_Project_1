import sys
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, mean_squared_log_error

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, RegressionMetricArtifact
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[float, object]: # type: ignore
        logging.info("Entered get_model_object_and_report method of ModelTrainer class")
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a RandomForestClassifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Training XGBRegressor with specified parameters")

            # Splitting the train and test data into features and target variables
            kf = KFold(n_splits=self.model_trainer_config.folds, shuffle=True, random_state=42)
            X = train[:, :-1]  # All columns except the last
            y = train[:, -1]
            pred = np.zeros(len(test))
            model = XGBRegressor(
                n_estimators=self.model_trainer_config.n_estimators,
                colsample_bytree=self.model_trainer_config.colsample_bytree,
                subsample=self.model_trainer_config.subsample,
                max_depth=self.model_trainer_config.max_depth,
                random_state=self.model_trainer_config.random_state,
                learning_rate=self.model_trainer_config.learning_rate,
                eval_metric=self.model_trainer_config.eval_metric
            )
            act_pred = test[:, -1]  # Actual target values for the test set
            logging.info("Entering KFold cross-validation")

            # Fit the model
            logging.info("Model training going on...")
            for i, (train_index, test_index) in enumerate(kf.split(X)):
                logging.info(f"Processing fold {i+1}")
                # Splitting the data into training and testing sets for each fold
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # Training the XGBRegressor model
                logging.info(f"Training model for fold {i+1}")
                model.fit(X_train, y_train,
                          eval_set=[(X_test, y_test)],
                          #early_stopping_rounds=100,
                          verbose=300)
                pred += model.predict(test[:, :-1])
                logging.info(f"Fold {i+1} completed.")
            logging.info("All folds processed successfully.")
            # Averaging the predictions across all folds
            pred /= self.model_trainer_config.folds
            logging.info("Model training done.")

            # Predictions and evaluation metrics
            _rmse = root_mean_squared_error(act_pred, pred)
            _r2 = r2_score(act_pred, pred)
            _rmsle = np.sqrt(mean_squared_log_error(act_pred, pred))

            # Creating metric artifact
            metric_artifact = RegressionMetricArtifact(rmse=_rmse, r2_score=_r2, rmsle=_rmsle) # type: ignore
            logging.info(f"Model metrics: RMSE={_rmse}, R2={_r2}, RMSLE={_rmsle}")
            return model, metric_artifact # type: ignore

        except Exception as e:
            raise MyException(e, sys) from e # type: ignore

    # def validate_data(self, arr):
    #     if isinstance(arr, np.ndarray):
    #         df = pd.DataFrame(arr)

    #         # Drop _id and id columns if present (by name or type)
    #         df = df.drop(columns=[col for col in df.columns if df[col].apply(type).eq(str).all()], errors='ignore') # type: ignore

    #         # Make sure there are still numeric columns left
    #         numeric_df = df.select_dtypes(include=[np.number])

    #         if numeric_df.shape[1] == 0:
    #             raise ValueError("After dropping non-numeric columns, no numeric features remain.")

    #         return numeric_df.to_numpy()

    #     return arr

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            # train_arr = self.validate_data(train_arr)
            # test_arr = self.validate_data(test_arr)
            logging.info("train-test data loaded")
            
            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            # Check if the model's accuracy meets the expected threshold
            # if mean_squared_log_error(train_arr[:, -1], trained_model.predict(train_arr[:, :-1])) > self.model_trainer_config.expected_accuracy: # type: ignore
            if metric_artifact.rmsle > self.model_trainer_config.expected_accuracy: # type: ignore
                logging.info("Model performance is not better than the expected score.")
                logging.info("No model found with score above the base score")
                raise Exception("No model found with score above the base score")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model) # type: ignore
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,# type: ignore
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e # type: ignore