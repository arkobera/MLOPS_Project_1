from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import root_mean_squared_log_error as rmsle
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass

@dataclass
class EvaluateModelResponse:
    trained_model_rmsle: float
    best_model_rmsle: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e # type: ignore[no-untyped-def]

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise  MyException(e,sys) # type: ignore[no-untyped-def]
        
    def _map_sex_column(self, df):
        """Map Sex column to 0 for Female and 1 for Male."""
        logging.info("Mapping 'Sex' column to binary values")
        #df['Sex'].fillna('Unknown', inplace=True)  # Fill NaN values with 'Unknown'
        df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
        return df

    # def _create_dummy_columns(self, df):
    #     """Create dummy variables for categorical features."""
    #     logging.info("Creating dummy variables for categorical features")
    #     df = pd.get_dummies(df, drop_first=True)
    #     return df

    # def _rename_columns(self, df):
    #     """Rename specific columns and ensure integer types for dummy columns."""
    #     logging.info("Renaming specific columns and casting to int")
    #     df = df.rename(columns={
    #         "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
    #         "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
    #     })
    #     for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
    #         if col in df.columns:
    #             df[col] = df[col].astype('int')
    #     return df
    
    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        if "id" in df.columns:
            df = df.drop("id", axis=1)
        return df

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")

            x = self._map_sex_column(x)
            x = self._drop_id_column(x)
            # x = self._create_dummy_columns(x)
            # x = self._rename_columns(x)

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_rmsle = self.model_trainer_artifact.metric_artifact.rmsle
            logging.info(f"RMSLE for this model: {trained_model_rmsle}")

            best_model_rmsle=None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing RMSLE for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_rmsle = rmsle(y, y_hat_best_model)
                logging.info(f"RMSLE-Production Model: {best_model_rmsle}, RMSLE-New Trained Model: {trained_model_rmsle}")

            if best_model_rmsle is None:
                is_model_accepted = True
                tmp_best_model_score = trained_model_rmsle
            else:
                is_model_accepted = trained_model_rmsle < best_model_rmsle
                tmp_best_model_score = best_model_rmsle

            result = EvaluateModelResponse(
                trained_model_rmsle=trained_model_rmsle,
                best_model_rmsle=best_model_rmsle, # type: ignore
                is_model_accepted=is_model_accepted, # type: ignore
                difference=abs(trained_model_rmsle - tmp_best_model_score) # type: ignore
            )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys) # type: ignore[no-untyped-def]

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e # type: ignore[no-untyped-def]