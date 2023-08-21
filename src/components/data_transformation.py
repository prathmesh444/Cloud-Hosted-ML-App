import os
import sys

import numpy as np
import pandas as pd

from dataclasses import dataclass


from src.Exception import CustomException
from src.logger import logging
import src.utils as utils


@dataclass()
class DataTransformationConfig:
    preprocessor_path: str = os.path.join(os.getcwd(), "artifacts", "preprocessor.pkl")


class DataTranformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_transformation(self, train_path, test_path):
        try:
            logging.info("Initiating Data Transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Data read successfully into csv')

            preprocessor = utils.get_data_transformer()
            logging.info('Preprocessor Created')

            target_col_name = "math_score"
            X_train = train_df.drop(columns=[target_col_name], axis=1)
            y_train = train_df[target_col_name]

            X_test = test_df.drop(columns=[target_col_name], axis=1)
            y_test = test_df[target_col_name]

            logging.info('Applying Preprocessor on Train Data')
            preprocessed_X_train = preprocessor.fit_transform(X_train)
            preprocessed_X_test = preprocessor.transform(X_test)

            train_arr = np.c_[preprocessed_X_train, np.array(y_train)]
            test_arr = np.c_[preprocessed_X_test, np.array(y_test)]
            logging.info('Preprocessor Applied on Train Data')

            logging.info('Saving Preprocessed Data')
            logging.info("Data Transformation Completed")
            utils.save_model(
                file_path=self.data_transformation_config.preprocessor_path,
                obj=preprocessor,
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
