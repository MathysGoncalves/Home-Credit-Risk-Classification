import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

#@Nikos Tavoularis
def missing_values_table(df:pd.DataFrame)-> pd.DataFrame:
    """
    Counts msising value and percentage of missing values in columns of the dataframe in decsending order
    
    Parameters
    ----------
    df
        pandas dataframe to count missing values.
 
    Returns
    -------
    dataframe with name of columns containing missing values, number of missing values, % of missing values
        pd.DataFrame
    """
    
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)
    #print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
    #    "There are " + str(mis_val_table_ren_columns.shape[0]) +
    #        " columns that have missing values.")
    #print(mis_val_table_ren_columns)
   
    return mis_val_table_ren_columns


def delete_missing_values_cols(df:pd.DataFrame, threshold:float=15)-> pd.DataFrame:
   """
   Deletes columns which % of missing values is higher than threshold specified by user
    
    Parameters
    ----------
    df
        train dataframe.
    threshold
        threshold of % of missing value defined by user 
 
    Returns
    -------
    dataframes with corresponding columns deleted
        pd.DataFrame,pd.DataFrame
   """
    
   missing_values_summary = missing_values_table(df)
   deleting_col_names = missing_values_summary[missing_values_summary['% of Total Values']>= threshold].index.values
   new_df=df.drop(deleting_col_names, axis=1, inplace=False)
   return new_df

#@Will Koehrsen
def numerizer(df:pd.DataFrame)->pd.DataFrame:
    """
    Function which encodes object type column created by Will Koehrsen

    Parameters
    ----------
    df_train
        train dataframe.
    df_test
        mtest dataframe.
 
    Returns
    -------
    dataframes with encoded columns
        pd.DataFrame,pd.DataFrame
    """
    
    # Create a label encoder object
    le = LabelEncoder()
    le_count = 0

    # Iterate through the columns
    for col in df:
        if df[col].dtype == 'object':
            # If 2 or fewer unique categories
            if len(list(df[col].unique())) <= 2:
                # Train on the training data
                le.fit(df[col])
                # Transform both training and testing data
                df[col] = le.transform(df[col])
                
                # Keep track of how many columns were label encoded
                le_count += 1
                
    # one-hot encoding of categorical variables
    df = pd.get_dummies(df)

    #print('Training Features shape: ', df_train.shape)
    #print('Testing Features shape: ', df_test.shape)

    return df
#@Will Koehrsen
def aligner(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """
    Function which aligns train dataframe and test dataframe to have same columns; inspired by Will Koehrsen

    Parameters
    ----------
    df
        train dataframe.
 
    Returns
    -------
    dataframes with aligned columns
        pd.DataFrames
    """
    train_labels = df_train['TARGET'] 
    # Align the training and testing data, keep only columns present in both dataframes
    df_train, df_test = df_train.align(df_test, join = 'inner', axis = 1)

    # Add the target back in
    df_train['TARGET'] = train_labels

    #print('Training Features shape: ', df_train.shape)
    #print('Testing Features shape: ', df_test.shape)
    return df_train,df_test
   
def missing_values_imputer(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Function which replaces missing values with median value of the column; inspired by Will Koehrsen

    Parameters
    ----------
    df_train
        train dataframe.
    df_test
        mtest dataframe.
 
    Returns
    -------
    dataframes with replaced missing values
        pd.DataFrames
    """
    train_columns=list(df_train.columns.values)
    train_columns.remove("TARGET")
    #print(train_columns)
    target = df_train['TARGET']
    df_train = df_train.drop(columns = ['TARGET'])
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    # Fit on the training data
    imputer.fit(df_train)

    # Transform both training
    train = imputer.transform(df_train)
    train_df = pd.DataFrame(train, columns = train_columns)
    train_df['TARGET']=target
    return train_df

def min_max_scaler(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function min max scales columns because of shown highly skewed column, and wide range numerics
    in several columns; inspired by Will Koehrsen

    Parameters
    ----------
    df
        train dataframe.
 
    Returns
    -------
    dataframes with replaced missing values
        pd.DataFrames
    """
    columns=list(df.columns.values)
    columns.remove("TARGET")
    columns.remove("SK_ID_CURR")

    target = df['TARGET']
    train_id = df['SK_ID_CURR']
    df = df.drop(columns = ['TARGET','SK_ID_CURR'])
    scaler = MinMaxScaler()

    # Fit on the training data
    scaler.fit(df)

    # Transform both training and testing data
    train = scaler.transform(df)

    df = pd.DataFrame(train, columns = columns)

    df['TARGET']=target
    df['SK_ID_CURR']=train_id.astype(int)
    return df
