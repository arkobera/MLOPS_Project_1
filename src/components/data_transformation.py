import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys) # type: ignore[return]

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys) # type: ignore[return]

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize transformers
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()
            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")

            # Load schema configurations
            num_features = self._schema_config['num_features']
            mm_columns = self._schema_config['mm_columns']
            logging.info("Cols loaded from schema.")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, num_features),
                    ("MinMaxScaler", min_max_scaler, mm_columns)
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e # type: ignore[return]

    def _map_sex_column(self, df):
        """Map Sex column to 0 for Female and 1 for Male."""
        logging.info("Mapping 'Sex' column to binary values")
        #df['Sex'].fillna('Unknown', inplace=True)  # Fill NaN values with 'Unknown'
        df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
        return df

    # def _encode_categorical_columns(self, df):
    #     """
    #     For each categorical column, create three encodings:
    #     1. Label encoding (sklearn)
    #     2. Factorize encoding (pandas)
    #     3. Mean encoding with respect to Heart_Rate and Body_Temp
    #     """
    #     logging.info("Encoding categorical columns with LabelEncoder, pd.factorize, and mean encoding.")
    #     categorical_columns = self._schema_config['categorical_columns']
    #     for col in categorical_columns:
    #         # 1. Label Encoding
    #         le = LabelEncoder()
    #         df[f"{col}_label"] = le.fit_transform(df[col])

    #         # 2. Factorize Encoding
    #         df[f"{col}_factorize"], _ = pd.factorize(df[col])

    #         # 3. Mean Encoding with respect to Heart_Rate and Body_Temp
    #         # For Heart_Rate
    #         # mean_map_hr = df.groupby(col)['Heart_Rate'].mean()
    #         # df[f"{col}_hr_mean"] = df[col].map(mean_map_hr)
    #         # # For Body_Temp
    #         # mean_map_bt = df.groupby(col)['Body_Temp'].mean()
    #         # df[f"{col}_bt_mean"] = df[col].map(mean_map_bt)
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
        drop_cols = self._schema_config['drop_columns']
        #print(df.head())
        df = df.drop(columns=drop_cols, errors='ignore')
        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            print(train_df.head())
            logging.info("Train-Test data loaded")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            # Apply custom transformations in specified sequence
            input_feature_train_df = self._map_sex_column(input_feature_train_df)
            input_feature_train_df = self._drop_id_column(input_feature_train_df)
            #input_feature_train_df = self._encode_categorical_columns(input_feature_train_df)
            #input_feature_train_df = self._rename_columns(input_feature_train_df)

            input_feature_test_df = self._map_sex_column(input_feature_test_df)
            input_feature_test_df = self._drop_id_column(input_feature_test_df)
            #input_feature_test_df = self._encode_categorical_columns(input_feature_test_df)
            #input_feature_test_df = self._rename_columns(input_feature_test_df)
            logging.info("Custom transformations applied to train and test data")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e # type: ignore[return]