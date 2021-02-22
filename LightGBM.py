import lightgbm
import optuna
import sklearn
import databases
import boosting_alg


def basic_LightGBM():
    db = databases.get_db("life expectancy")
    X_train = db.X_train
    y_train = db.y_train
    X_test = db.X_test
    y_test = db.y_test
    lightgbm_model = lightgbm.LGBMRegressor()
    lightgbm_model.fit(X_train, y_train)
    boosting_alg.training_results(lightgbm_model,X_train, y_train)
    boosting_alg.testing_results(lightgbm_model,X_test, y_test)

    return

def objective(trial):
    db = databases.get_db("life expectancy")
    X_train = db.X_train
    y_train = db.y_train
    num_leaves = int(trial.suggest_loguniform('num_leaves', 20, 100))
    n_estimators = int(trial.suggest_loguniform('n_estimators', 1, 100))
    max_depth = int(trial.suggest_loguniform('max_depth', 1, 50))
    subsample_for_bin = int(trial.suggest_loguniform('subsample_for_bin', 150000, 200000))
    clf = lightgbm.LGBMRegressor(num_leaves=num_leaves, max_depth=max_depth, n_estimators=n_estimators,
                                 subsample_for_bin=subsample_for_bin)
    return sklearn.model_selection.cross_val_score(clf, X_train, y_train,
                                                   n_jobs=-1, cv=3).mean()

def optimize_hyper_parameters_dt():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)
    trial = study.best_trial
    print('Accuracy: {}'.format(trial.value))
    print("Best hyperparameters: {}".format(trial.params))

def LightGBM_comparision():
    db = databases.get_db("life expectancy")
    X_train = db.X_train
    y_train = db.y_train
    X_test = db.X_test
    y_test = db.y_test
    # life exp. best: num_leaves = 34, n_est = 89, max_depth = 2, subsamples for bin = 188219
    complex_LightGBM_model = lightgbm.LGBMRegressor(num_leaves=34, max_depth=2, n_estimators=89,
                                 subsample_for_bin=188219)
    complex_LightGBM_model.fit(X_train, y_train)
    print("basic:")
    basic_LightGBM()
    print("complex")
    boosting_alg.training_results(complex_LightGBM_model,X_train, y_train)
    boosting_alg.testing_results(complex_LightGBM_model,X_test, y_test)