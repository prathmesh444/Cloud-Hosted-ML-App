import sys
from src.Exception import CustomException
from src.logger import logging

import pandas as pd
class Predict_Pipeline():
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Initiating Prediction")
            model = pd.read_pickle("src/components/artifacts/model.pkl")
            preprocessing = pd.read_pickle("src/components/artifacts/preprocessor.pkl")
            features = preprocessing.transform(features)
            logging.info("Model Loaded Successfully")
            prediction = model.predict(features)
            logging.info("Prediction Completed")
            return prediction
        except Exception as e:
            raise CustomException(e, sys)

class CustomData():
    def __init__(self, gender, race, parental_edu, lunch, test_prep_course, reading_score, writing_score):
        self.gender = gender
        self.race = race
        self.parental_edu = parental_edu
        self.lunch = lunch
        self.test_prep_course = test_prep_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_DataFrame(self):
        try:
            temp_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race],
                "parental_level_of_education": [self.parental_edu],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_prep_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }
            return pd.DataFrame(temp_dict)
        except Exception as e:
            raise CustomException(e, sys)
