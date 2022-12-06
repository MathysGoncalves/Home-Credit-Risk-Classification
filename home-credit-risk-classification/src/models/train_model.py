from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
import lightgbm as lgb
import pandas as pd
import re
import time

def splitter(df:pd.DataFrame, test_size:float=0.2)->tuple:
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    X_train = X_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    X_test = X_test.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    return X_train,X_test,y_train,y_test

def trainer(X_train,y_train):
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

    parameters = {'num_leaves':[5,10,20,40,60,80,100],
                'n_estimators': [100,500,1500],
                'min_child_samples':[5,10,15],
                'max_depth':[-1,5,10,20],
                'learning_rate':[0.005,0.05,0.1,0.2],
                'reg_alpha':[0,0.01,0.03]}

    start = time.time()

    lgbm = lgb.LGBMClassifier()
    clf = RandomizedSearchCV(lgbm, parameters, scoring='f1_micro', cv=StratifiedKFold(n_splits=5), random_state=42, n_jobs=3)

    clf.fit(X_train, y_train)

    end = time.time()

    print('Execution time is:')
    print(end - start)
    print(clf.best_params_)
    return clf

