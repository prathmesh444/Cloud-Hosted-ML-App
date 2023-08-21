import os
import sys
from src.Exception import CustomException
from src.components.model_trainer import ModelTrainer
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTranformation
from src.components.data_transformation import DataTransformationConfig


@dataclass()
class DataIngestionConfig:
    train_data_path: str = os.path.join(os.getcwd(), "artifacts", "train.csv")
    test_data_path: str = os.path.join(os.getcwd(), "artifacts", "test.csv")
    raw_data_path: str = os.path.join(os.getcwd(), "artifacts", "raw_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_Ingestion(self):
        try:
            logging.info("Initiating Data Ingestion")

            df = pd.read_csv('C:/Users/hp/PycharmProjects/myProject/notebook/data/stud.csv')
            logging.info('Data read successfully into csv')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train-test split Initiated')
            train, test = train_test_split(df, test_size=0.2, random_state=42)
            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Train-test split Completed')
            logging.info("Data Ingestion Completed")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e)


if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_Ingestion()
    data_trans = DataTranformation()
    train_arr, test_arr, _ = data_trans.initiate_transformation(train_path, test_path)

    trainer = ModelTrainer()
    output = trainer.initiate_model_trainer(train_arr, test_arr)
    print(output)
