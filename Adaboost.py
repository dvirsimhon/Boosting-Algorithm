from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import optuna
import sklearn
import databases
import boosting_alg


def basic_Adaboost():
    db = databases.get_db("car prices")
    X_train = db.X_train
    y_train = db.y_train
    X_test = db.X_test
    y_test = db.y_test
    adaboost_model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=1))
    adaboost_model.fit(X_train, y_train)
    boosting_alg.training_results(adaboost_model,X_train, y_train)
    boosting_alg.testing_results(adaboost_model,X_test, y_test)

    return

def objective(trial):
    db = databases.get_db("car prices")
    X_train = db.X_train
    y_train = db.y_train
    n_estimators = int(trial.suggest_loguniform('n_estimators', 1, 100))
    learning_rate = float(trial.suggest_loguniform('learning_rate', 0.01, 1))
    clf = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=1),
                            n_estimators=n_estimators,learning_rate=learning_rate)
    return sklearn.model_selection.cross_val_score(clf, X_train, y_train,
                                                   n_jobs=-1, cv=3).mean()

def optimize_hyper_parameters_dt():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    trial = study.best_trial
    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

def Adaboost_comparision():
    db = databases.get_db("car prices")
    X_train = db.X_train
    y_train = db.y_train
    X_test = db.X_test
    y_test = db.y_test
    # life exp. best: n_est = 28, learning rate = 0.7
    complex_Adaboost_model = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=1),
                                           n_estimators=60, learning_rate=0.02)
    complex_Adaboost_model.fit(X_train, y_train)
    print("basic:")
    basic_Adaboost()
    print("complex")
    boosting_alg.training_results(complex_Adaboost_model,X_train, y_train)
    boosting_alg.testing_results(complex_Adaboost_model,X_test, y_test)