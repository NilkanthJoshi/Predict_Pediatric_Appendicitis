import os

# Project-specific configurations
AWS_S3_BUCKET_NAME = "pediatric-appendicitis-detection"
MONGO_DATABASE_NAME = "Major_Project"
MONGO_COLLECTION_NAME = "Data"

# Target column (adjust if there's a specific target for prediction)
TARGET_COLUMN = "Diagnosis"

# MongoDB connection URL
MONGO_DB_URL = "mongodb+srv://ronakbediya:rb12345@cluster0.swlu6.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Model-related configurations
MODEL_FILE_NAME = "appendicitis_model"
MODEL_FILE_EXTENSION = ".pkl"

# Folder for storing artifacts
artifact_folder = "artifacts"
