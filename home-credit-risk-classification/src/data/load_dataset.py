import pandas as pd
import numpy as np

def datasets_loader(train_dataset_path:str,test_dataset_path:str,percentage:int) -> tuple:
    """
    Loads dataset according to perecentage specified by the user
    
    Parameters
    ----------
    train_dataset_path
        path of training dataset.
    test_dataset_path
        path of testing dataset.
    percentage
        percentage of dataset that the user wants to be loaded

    Returns
    -------
    both loaded training datasets
        tuple
    """
    n_lines_train_file = 307511
    nlinesrandomsample_train = int(round(n_lines_train_file*(percentage/100),0))
    n_lines_test_file = 48744
    nlinesrandomsample_test = int(round(n_lines_test_file*(percentage/100),0))
    train_lines_2_skip = np.random.choice(np.arange(1,n_lines_train_file+1), (n_lines_train_file-nlinesrandomsample_train), replace=False)
    test_lines_2_skip = np.random.choice(np.arange(1,n_lines_test_file+1), (n_lines_test_file-nlinesrandomsample_test), replace=False)
    original_training_dataframe = pd.read_csv(train_dataset_path, skiprows=train_lines_2_skip)
    original_testing_dataframe = pd.read_csv(test_dataset_path, skiprows=test_lines_2_skip)
    return original_training_dataframe,original_testing_dataframe


