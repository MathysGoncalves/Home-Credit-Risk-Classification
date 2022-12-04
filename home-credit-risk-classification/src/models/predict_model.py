from sklearn.ensemble import RandomForestClassifier

def predicter(fited_model,X_test):
    predictions=fited_model.predict(X_test)
    return predictions