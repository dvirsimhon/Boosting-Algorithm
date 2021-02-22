import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

def training_results(model,X_train,y_train):
    print("Training: ")
    predictions = model.predict(X_train)
    score = metrics.r2_score(y_train, predictions)
    rmse = metrics.mean_squared_error(y_train, predictions, squared=False)
    print("R Squared : %s" % "{0:.5}".format(score))
    print("RMSE: %s" % "{0:.3}".format(rmse))


def testing_results(model,X_test,y_test):
    print("Testing: ")
    predictions = model.predict(X_test)
    score = metrics.r2_score(y_test, predictions)
    rmse = metrics.mean_squared_error(y_test, predictions)
    print("R Squared : %s" % "{0:.3}".format(score))
    print("RMSE: %s" % "{0:.3}".format(rmse))


def run_k_fold(tree_model,X_train,y_train):
    TRAIN = X_train.copy()
    TRAIN['class'] = y_train
    kf = KFold(n_splits=10)
    accuracy = []
    for train, test in kf.split(TRAIN):
        train_predictors = (X_train.iloc[train, :])
        train_target = TRAIN['class'].iloc[train]
        tree_model.fit(train_predictors, train_target)
        accuracy.append(tree_model.score(X_train.iloc[test, :], TRAIN['class'].iloc[test]))
    return np.mean(accuracy)
