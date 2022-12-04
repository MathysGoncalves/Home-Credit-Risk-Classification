from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import pandas as pd

def splitter(df:pd.DataFrame,test_size:float)->tuple:
    X=df.drop(columns=['TARGET','SK_ID_CURR'])
    y=df['TARGET']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train,X_test,y_train,y_test

def trainer(X_train,y_train,max_depth,random_state):

    clf = RandomForestClassifier(max_depth=max_depth, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf

