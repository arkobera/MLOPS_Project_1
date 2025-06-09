# from src.logger import logging
# from src.exception import MyException, error_message_detail
# import sys

# logging.info("Logging is configured successfully.")

# try:
#     a = 1 / 0
# except Exception as e:
#     error_message = error_message_detail(e, sys) # type: ignore
#     #logging.error(f"An error occurred: {error_message}")


from src.pipline.training_pipeline import TrainPipeline

if __name__ == "__main__":
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
    except Exception as e:
        print(f"An error occurred: {e}")
        # You can also log the error or handle it as needed


# workflow 
# Constants --> ConfigEntity --> ArtifactEntity --> Components --> Pipeline