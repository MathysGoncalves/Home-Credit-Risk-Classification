import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from data.load_dataset import datasets_loader
from features import build_features
from models import train_model
from models import predict_model

import logging



logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    acc_score = accuracy_score(actual, pred)
    return rmse, mae, r2, acc_score


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    percentage=float(sys.argv[1])
    max_depth = float(sys.argv[2])
    random_state = int(sys.argv[3])
    y_test_size= float(sys.argv[4])
    delete_threshold= float(sys.argv[5])

    train_dataset_path,test_dataset_path='./data/application_train.csv','./data/application_test.csv'

    # Read the csv file
    try:
        train_dataset,test_dataset = datasets_loader(train_dataset_path,test_dataset_path, percentage)
    except Exception as e:
        logger.exception(
            "Unable to read files check path or arguments Error: %s", e
        )
    
    #building features/preprocessing
    train_dataframe,test_dataframe= build_features.delete_missing_values_cols(train_dataset,test_dataset,delete_threshold)
    train_dataframe,test_dataframe= build_features.numerizer(train_dataframe, test_dataframe)
    train_dataframe,test_dataframe= build_features.aligner(train_dataframe, test_dataframe)
    train_dataframe,test_dataframe= build_features.missing_values_imputer(train_dataframe, test_dataframe)
    train_dataframe,test_dataframe= build_features.min_max_scaler(train_dataframe, test_dataframe)
    
    # Split the data into training and test sets
    train_x,test_x,train_y,test_y = train_model.splitter(train_dataframe,y_test_size)



    with mlflow.start_run():
        model = train_model.trainer(train_x,train_y,max_depth,random_state)
        predictions= predict_model.predicter(model,test_x)

        (rmse, mae, r2, acc_score) = eval_metrics(test_y, predictions)

        print("RandomForestClassifier model (max_depth=%f, random_state=%f):" % (max_depth, random_state))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        print("  Accuracy_score: %s" % acc_score)

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("Accuracy_score", acc_score)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name="RandomForestClassiferModel")
        else:
            mlflow.sklearn.log_model(model, "model")
