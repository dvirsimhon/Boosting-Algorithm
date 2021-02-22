import catboost
import optuna
import sklearn
import databases
import boosting_alg


def basic_CatBoost():
    db = databases.get_db("car prices")
    X_train = db.X_train
    y_train = db.y_train
    X_test = db.X_test
    y_test = db.y_test
    catboost_model = catboost.CatBoostRegressor()
    catboost_model.fit(X_train, y_train)
    boosting_alg.training_results(catboost_model,X_train, y_train)
    boosting_alg.testing_results(catboost_model,X_test, y_test)

    return

def objective(trial):
    db = databases.get_db("car prices")
    X_train = db.X_train
    y_train = db.y_train
    max_depth = int(trial.suggest_loguniform('depth', 1, 16))
    n_estimators = int(trial.suggest_loguniform('n_estimators', 1, 100))
    learning_rate = float(trial.suggest_loguniform('learning_rate', 0.001, 1))
    clf = catboost.CatBoostRegressor(learning_rate=learning_rate,
                                        max_depth=max_depth, n_estimators=n_estimators)
    return sklearn.model_selection.cross_val_score(clf, X_train, y_train,
                                                   n_jobs=-1, cv=3).mean()

def optimize_hyper_parameters_dt():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    trial = study.best_trial
    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

def CatBoost_comparision():
    db = databases.get_db("car prices")
    X_train = db.X_train
    y_train = db.y_train
    X_test = db.X_test
    y_test = db.y_test
    # life exp. best: learning rate = 0.974, max depth = 3, n estimators = 57
    # car prices best: learning rate = 0.521, max depth = 8, n_estimators = 51
    complex_CatBoost_model = catboost.CatBoostRegressor(learning_rate=0.974,
                                        max_depth=3, n_estimators=57)

    complex_CatBoost_model.fit(X_train, y_train)
    print("basic:")
    basic_CatBoost()
    print("complex")
    boosting_alg.training_results(complex_CatBoost_model,X_train, y_train)
    boosting_alg.testing_results(complex_CatBoost_model,X_test, y_test)