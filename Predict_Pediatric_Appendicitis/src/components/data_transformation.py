import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    artifact_dir = os.path.join(artifact_folder)
    transformed_train_file_path = os.path.join(artifact_dir, "train.npy")
    transformed_test_file_path = os.path.join(artifact_dir, "test.npy")
    transformed_object_file_path = os.path.join(artifact_dir, "preprocessor.pkl")


class DataTransformation:
    def __init__(self, feature_store_file_path):
        self.feature_store_file_path = feature_store_file_path
        self.data_transformation_config = DataTransformationConfig()
        self.utils = MainUtils()

    @staticmethod
    def get_data(feature_store_file_path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(feature_store_file_path)
            data.rename(columns={"Diagnosis": TARGET_COLUMN}, inplace=True)
            return data
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self, X: pd.DataFrame):
        """
        Create a preprocessing pipeline for numeric and categorical features.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            ColumnTransformer: Preprocessing pipeline.
        """
        try:
            # Identify numeric and categorical columns
            numeric_cols = X.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

            # Preprocessing for numeric features
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", RobustScaler()),
                ]
            )

            # Preprocessing for categorical features
            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            )

            # Combine the transformations using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_cols),
                    ("cat", categorical_transformer, categorical_cols),
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        logging.info(
            "Entered initiate_data_transformation method of DataTransformation class"
        )

        try:
            # Load the data from the feature store
            dataframe = self.get_data(
                feature_store_file_path=self.feature_store_file_path
            )

            # Separate features (X) and target (y)
            X = dataframe.drop(columns=TARGET_COLUMN)
            y = dataframe[TARGET_COLUMN].apply(
                lambda x: 1 if x == "appendicitis" else 0
            )

            # Split the data into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Get the preprocessing pipeline
            preprocessor = self.get_data_transformer_object(X_train)

            # Apply the preprocessing steps to the training and test data
            X_train_scaled = preprocessor.fit_transform(X_train)
            X_test_scaled = preprocessor.transform(X_test)

            # Save the preprocessor object to a file
            preprocessor_path = (
                self.data_transformation_config.transformed_object_file_path
            )
            os.makedirs(os.path.dirname(preprocessor_path), exist_ok=True)
            self.utils.save_object(file_path=preprocessor_path, obj=preprocessor)

            # Combine the scaled features with the target variable
            train_arr = np.c_[X_train_scaled, np.array(y_train)]
            test_arr = np.c_[X_test_scaled, np.array(y_test)]

            # Return the transformed training, test arrays, and the preprocessor path
            return train_arr, test_arr, preprocessor_path

        except Exception as e:
            raise CustomException(e, sys) from e
