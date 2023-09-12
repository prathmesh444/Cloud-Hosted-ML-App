import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from src.Exception import CustomException
from src.logger import logging
from src.utils import save_model
from src.utils import evaluate_models


@dataclass()
class ModelTrainerConfig:
    def __init__(self):
        self.train_model_path = os.path.join(os.getcwd(), "artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("splitting train and test data")
            X_train, y_train, X_test, y_test = [
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            ]
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "Ada Boost": AdaBoostRegressor(),
                "Gradient Boost": GradientBoostingRegressor(),
                "KNN": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
            }

            # For hyperparameter tuning these parameter are created
            params = {
                'Linear Regression': {},
                'Decision Tree': {
                    'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    # 'splitter': ['best', 'random'],
                },
                'Random Forest': {
                    # 'criterion': ['mse', 'friedman_mse', 'mae', 'poisson'],
                    # 'splitter': ['best', 'random'],
                    'n_estimators': [8, 16, 24, 32, 64, 128, 256, 512],
                },
                'Ada Boost': {
                    'learning rate': [0.01, 0.05, 0.1, 0.3, 1],
                    # 'loss': ['linear', 'square', 'exponential'],
                    'n_estimators': [8, 16, 24, 32, 64, 128, 256, 512],
                },
                'Gradient Boost': {
                    'n_estimators': [8, 16, 24, 32, 64, 128, 256, 512],
                    'learning rate': [0.01, 0.05, 0.1, 0.3, 1],
                    # 'loss': ['ls', 'lad', 'huber', 'quantile'],
                    # 'max_features': ['auto', 'sqrt', 'log2'],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                },
                'KNN': {
                    'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9],
                    # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    # 'weights': ['uniform', 'distance'],
                },
                'XGBoost': {
                    'n_estimators': [8, 16, 24, 32, 64, 128, 256, 512],
                    'learning rate': [0.01, 0.05, 0.1, 0.3, 1],
                },
                'CatBoost': {
                    'iterations': [8, 16, 24, 32, 64, 128, 256, 512],
                    'learning rate': [0.01, 0.05, 0.1, 0.3, 1],
                    'depth': [2, 3, 4, 5],
                }
            }
            trained_models, trained_models_score = evaluate_models(X_train, y_train, X_test, y_test,
                                                                   models=models, params=params)
            best_model_index = trained_models_score.index(max(trained_models_score))
            best_model = trained_models[best_model_index]
            best_model_score = trained_models_score[best_model_index]

            if (best_model_score < 0.6):
                raise CustomException("Model Score is less than 0.6, \n No Best Model Found", sys)

            save_model(
                file_path=self.model_trainer_config.train_model_path,
                obj=best_model,
            )
            pred_y_test = best_model.predict(X_test)
            pred_r2_score = r2_score(y_test, pred_y_test)

            return pred_r2_score
        except Exception as e:
            raise CustomException(e, sys)
