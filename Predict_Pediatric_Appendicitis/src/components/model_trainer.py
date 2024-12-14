import sys
import os
import numpy as np
from typing import Tuple, Dict
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join("artifacts")
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy = 0.6
    model_config_file_path = os.path.join("config", "model.yaml")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = MainUtils()
        self.models = {
            "XGBClassifier": XGBClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "SVC": SVC(),
            "RandomForestClassifier": RandomForestClassifier(),
        }

    def evaluate_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate all models and return their accuracy scores."""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            report = {}

            for model_name, model in self.models.items():
                model.fit(X_train, y_train)
                y_test_pred = model.predict(X_test)
                test_model_score = accuracy_score(y_test, y_test_pred)
                report[model_name] = test_model_score

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def get_best_model(self, model_report: Dict[str, float]) -> Tuple[str, object]:
        """Determine the best model based on accuracy scores."""
        best_model_name = max(model_report, key=model_report.get)
        best_model_object = self.models[best_model_name]
        best_model_score = model_report[best_model_name]

        return best_model_name, best_model_object, best_model_score

    def finetune_best_model(
        self,
        best_model_object: object,
        best_model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> object:
        """Fine-tune the best model using grid search."""
        try:
            model_param_grid = self.utils.read_yaml_file(
                self.model_trainer_config.model_config_file_path
            )["model_selection"]["model"][best_model_name]["search_param_grid"]
            grid_search = GridSearchCV(
                best_model_object,
                param_grid=model_param_grid,
                cv=5,
                n_jobs=-1,
                verbose=1,
            )
            grid_search.fit(X_train, y_train)

            best_params = grid_search.best_params_
            logging.info("Best parameters found: %s", best_params)

            finetuned_model = best_model_object.set_params(**best_params)
            return finetuned_model

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(
        self, train_array: np.ndarray, test_array: np.ndarray
    ) -> str:
        """Main method to initiate model training."""
        try:
            logging.info("Extracting training and testing features")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            logging.info("Evaluating models")
            model_report = self.evaluate_models(X_train, y_train)

            best_model_name, best_model, _ = self.get_best_model(model_report)
            logging.info("Best model found: %s", best_model_name)

            best_model = self.finetune_best_model(
                best_model, best_model_name, X_train, y_train
            )
            best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)
            best_model_score = accuracy_score(y_test, y_pred)

            logging.info(f"Best model: {best_model_name}, Score: {best_model_score}")

            if best_model_score < self.model_trainer_config.expected_accuracy:
                raise Exception(
                    f"No best model found with accuracy greater than {self.model_trainer_config.expected_accuracy}"
                )

            logging.info(
                "Saving the model at: %s", self.model_trainer_config.trained_model_path
            )
            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_path),
                exist_ok=True,
            )
            self.utils.save_object(
                file_path=self.model_trainer_config.trained_model_path, obj=best_model
            )

            return self.model_trainer_config.trained_model_path

        except Exception as e:
            raise CustomException(e, sys)
