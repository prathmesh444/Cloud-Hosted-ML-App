import os
import sys
import dill

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.Exception import CustomException
from src.logger import logging


def get_data_transformer():
    try:
        categorical_feature = ["gender", "race_ethnicity", "parental_level_of_education", "lunch",
                               "test_preparation_course"]
        numerical_feature = ["reading_score", "writing_score"]

        num_pipeline = Pipeline(
            steps=[
                ("std_scaler", StandardScaler(with_mean=False)),
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        cat_pipeline = Pipeline(
            steps=[
                ("one_hot_encoder", OneHotEncoder()),
                ("std_scaler", StandardScaler(with_mean=False)),
                ("imputer", SimpleImputer(strategy="most_frequent")),
            ]
        )
        logging.info('Categorical and Numerical Pipeline Created')

        preprocessor = ColumnTransformer(
            [
                ("numerical", num_pipeline, numerical_feature),
                ("categorical", cat_pipeline, categorical_feature),
            ]
        )
        logging.info('Preprocessor Created')

        return preprocessor

    except Exception as e:
        raise CustomException(e, sys)


def save_model(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        trained_models = []
        trained_models_score = []
        for model_name, model in models.items():
            para = params[model_name]

            # hyperparameter tuning, GridSearchCV cross-validation
            GS = GridSearchCV(model, para, cv=3)
            GS.fit(X_train, y_train)

            model.set_params(**GS.best_params_)
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_score = r2_score(y_test, y_test_pred)

            trained_models.append(model)
            trained_models_score.append(test_score)

        return trained_models, trained_models_score

    except Exception as e:
        raise CustomException(e, sys)