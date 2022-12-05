from sklearn.ensemble import RandomForestClassifier

def predicter(fited_model,X_test):
    """
    Function which predicts classes with given model and test_x

    Parameters
    ----------
    fited_model
        sklearn model
    X_test
        List
        test set
    Returns
    -------
    List predictions
        List
    """
    predictions=fited_model.predict(X_test)
    return predictions