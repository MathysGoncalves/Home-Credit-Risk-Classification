from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

def splitter(df:pd.DataFrame,test_size:float)->tuple:
    """
    Function which splits dataframe into train and test sets

    Parameters
    ----------
    df
        dataframe to be splitted
    test_size
        float (>0 and <1)
        Specifies the test set size to be splitted
    Returns
    -------
    Lists train_x,test_x,train_y,test_y
        Lists
    """
    X=df.drop(columns=['TARGET','SK_ID_CURR'])
    y=df['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train,X_test,y_train,y_test

def trainer(X_train,y_train,max_depth,random_state):
    """
    Function which trains model on train_x and train_y

    Parameters
    ----------
    max_depth
        int, default=None
        The maximum depth of the tree. 
        If None, then nodes are expanded until all leaves are pure 
        or until all leaves contain less than min_samples_split samples.

    random_state
        int, RandomState instance or None, default=None
        Controls both the randomness of the bootstrapping 
        of the samples used when building trees (if bootstrap=True) 
        and the sampling of the features to consider when 
        looking for the best split at each node (if max_features < n_features)
    Returns
    -------
    Model clf
        sklearn model
    """
    clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf

