import sys
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from zipfile import Path
from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join(artifact_folder)


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtils()

    def export_collection_as_dataframe(self, collection_name, db_name):
        """
        This function connects to MongoDB, fetches the collection,
        and converts it to a DataFrame.
        """
        try:
            mongo_client = MongoClient(MONGO_DB_URL)
            collection = mongo_client[db_name][collection_name]

            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def export_data_into_feature_store_file_path(self) -> pd.DataFrame:
        """
        This function exports the data from MongoDB into a CSV file in the feature store.
        """
        try:
            logging.info(f"Exporting data from MongoDB")
            raw_file_path = self.data_ingestion_config.artifact_folder

            os.makedirs(raw_file_path, exist_ok=True)

            # Fetching the data from MongoDB
            pediatric_data = self.export_collection_as_dataframe(
                collection_name=MONGO_COLLECTION_NAME, db_name=MONGO_DATABASE_NAME
            )

            logging.info(
                f"Saving exported data into feature store file path: {raw_file_path}"
            )

            # Save the data into CSV file
            feature_store_file_path = os.path.join(
                raw_file_path, "pediatric_appendicitis_data.csv"
            )
            pediatric_data.to_csv(feature_store_file_path, index=False)

            return feature_store_file_path

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> Path:
        """
        Initiates the data ingestion process, exporting the data into a feature store path.
        """
        logging.info(
            "Entered initiate_data_ingestion method of the data ingestion class"
        )

        try:
            feature_store_file_path = self.export_data_into_feature_store_file_path()

            logging.info("Data successfully retrieved from MongoDB")

            logging.info(
                "Exited initiate_data_ingestion method of the data ingestion class"
            )

            return feature_store_file_path
        except Exception as e:
            raise CustomException(e, sys) from e
