import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score
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
    """
    Function which calculates metrics all at once

    Parameters
    ----------
    actual
        List
        True classes
    pred
        List
        Predicted classes
    Returns
    -------
    Floats rmse, mae, r2, acc_score
        Floats
    """
    roc_auc = roc_auc_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    accuracy = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred)
    return roc_auc, precision, recall, accuracy, f1


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # Variables to be specified while calling the main.py file 
    percentage = int(sys.argv[1])             #% of the dataset you want to read
    #max_depth = float(sys.argv[2])             #max_depth of the random forest classifier model you will train
    #random_state = int(sys.argv[3])            #random_state of the random forest classifier model you will train
    #y_test_size = float(sys.argv[4])           #y_test_size as the size of test set for the train_test split
    #delete_threshold = float(sys.argv[2])      #delete_threshold threshold which specifies the % of missing values from which 
                                               # you will delete columns
    #-------------------------------------------------------------------------------

    #! CHANGE THESE PATHS TO YOUR CONVENIENCE IF NEEDED
    train_dataset_path,test_dataset_path='home-credit-risk-classification/data/application_train.csv','./home-credit-risk-classification/data/application_test.csv' 
    #train_url = 'https://drive.google.com/file/d/1dqswp6BOPyb86kF8bmkEt72AQVjsRDm3/view?usp=sharing'
    #train_dataset_path = 'https://drive.google.com/uc?export=download&id='+train_url.split('/')[-2]
    #test_url='https://drive.google.com/file/d/1ffdLDLekINEKcofjEkuaJE-qlHeJ51Ci/view?usp=sharing'
    #test_dataset_path = 'https://drive.google.com/uc?export=download&id='+test_url.split('/')[-2]

    # Read the csv file
    try:
        train_dataset = datasets_loader(train_dataset_path, percentage)
    except Exception as e:
        logger.exception(
            "Unable to read files check path or arguments Error: %s", e
        )
    
    #building features/preprocessing
    train_dataframe = build_features.delete_missing_values_cols(train_dataset)
    train_dataframe = build_features.numerizer(train_dataframe)
    #train_dataframe,test_dataframe= build_features.aligner(train_dataframe, test_dataframe)
    train_dataframe = build_features.missing_values_imputer(train_dataframe)
    train_dataframe = build_features.min_max_scaler(train_dataframe)
    
    # Split the data into training and test sets
    train_x,test_x,train_y,test_y = train_model.splitter(train_dataframe)


    with mlflow.start_run():
        model = train_model.trainer(train_x,train_y)
        predictions= predict_model.predicter(model,test_x)

        (roc_auc, precision, recall, acc_score, f1) = eval_metrics(test_y, predictions)

        print("LightGBM model (max_depth=%f, random_state=%f):")
        print("  roc_auc: %s" % roc_auc)
        print("  Precision %s" % precision)
        print("  Recall: %s" % recall)
        print("  F1: %s" % f1)
        print("  Accuracy_score: %s" % acc_score)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("Accuracy_score", acc_score)
        mlflow.log_metric("f1", f1)

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
