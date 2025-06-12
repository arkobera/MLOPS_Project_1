import sys
from src.entity.config_entity import CaloriePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame


class CalorieData:
    def __init__(self,
                Sex,
                Age,
                Height,
                Weight,
                Duration,
                Heart_Rate,
                Body_Temp,
                ):
        """
        Vehicle Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Sex = Sex
            self.Age = Age
            self.Height = Height
            self.Weight = Weight
            self.Duration = Duration
            self.Heart_Rate = Heart_Rate
            self.Body_Temp = Body_Temp

        except Exception as e:
            raise MyException(e, sys) from e # type: ignore

    def get_calorie_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from CalorieData class input
        """
        try:

            calorie_input_dict = self.get_calorie_data_as_dict()
            return DataFrame(calorie_input_dict)

        except Exception as e:
            raise MyException(e, sys) from e # type: ignore


    def get_calorie_data_as_dict(self):
        """
        This function returns a dictionary from CalorieData class input
        """
        logging.info("Entered get_calorie_data_as_dict method as CalorieData class")

        try:
            input_data = {
                "Sex": [self.Sex],
                "Age": [self.Age],
                "Height": [self.Height],
                "Weight": [self.Weight],
                "Duration": [self.Duration],
                "Heart_Rate": [self.Heart_Rate],
                "Body_Temp": [self.Body_Temp],
            }

            logging.info("Created calorie data dict")
            logging.info("Exited get_calorie_data_as_dict method as CalorieData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e # type: ignore

class CalorieDataRegressor:
    def __init__(self,prediction_pipeline_config: CaloriePredictorConfig = CaloriePredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys) # type: ignore

    def predict(self, dataframe) -> str:
        """
        This is the method of CalorieDataRegressor
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of CalorieDataRegressor class")
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result # type: ignore
        
        except Exception as e:
            raise MyException(e, sys) # type: ignore