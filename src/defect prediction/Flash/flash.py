__author__ = 'Fahmid'

import random

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.tree import DecisionTreeRegressor

from tuner import DT_TUNER
import numpy as np

from measure import get_counts, calculate_average_odds_difference

BUDGET = 20
POOL_SIZE = 10000
INIT_POOL_SIZE = 10

def tune_dt(x_train, y_train, project_name):
    tuner = DT_TUNER()
    sss = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=0)
    for train_index, tune_index in sss.split(x_train, y_train):
        x_train_flash, x_tune_flash = x_train[train_index], x_train[tune_index]
        y_train_flash, y_tune_flash = y_train.iloc[train_index], y_train.iloc[tune_index]
        best_conf = tune_with_flash(tuner, x_train_flash, y_train_flash, x_tune_flash, y_tune_flash, project_name,
                                    random_seed=1)

    return best_conf


def tune_with_flash(tuner, x_train, y_train, x_tune, y_tune, project_name, test_df, biased_col, random_seed=0):
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    random.seed(random_seed)

    print("DEFAULT AOD: " + str(measure_fitness(tuner, x_train, y_train, x_tune, y_tune, tuner.default_config, test_df, biased_col)))

    this_budget = BUDGET

    # Make initial population
    param_search_space = tuner.generate_param_pools(POOL_SIZE)

    # Evaluate initial pool
    evaluted_configs = random.sample(param_search_space, INIT_POOL_SIZE)
    #param_search_space = list(set(param_search_space) - (set(evaluted_configs)))

    aod_scores = [measure_fitness(tuner, x_train, y_train, x_tune, y_tune, configs, test_df, biased_col, metric='aod') for configs in evaluted_configs]
    eod_scores = [measure_fitness(tuner, x_train, y_train, x_tune, y_tune, configs, test_df, biased_col, metric='eod') for configs in evaluted_configs]
    d2h_scores = [measure_fitness(tuner, x_train, y_train, x_tune, y_tune, configs, test_df, biased_col, metric='d2h')
                  for configs in evaluted_configs]
    # Filtering NaN case
    evaluted_configs, aod_scores = filter_no_info(project_name, evaluted_configs, aod_scores)
    evaluted_configs, eod_scores = filter_no_info(project_name, evaluted_configs, eod_scores)
    evaluted_configs, d2h_scores = filter_no_info(project_name, evaluted_configs, d2h_scores)


    print(project_name + " | AOD Score of init pool: " + str(aod_scores))
    print(project_name + " | EOD Score of init pool: " + str(eod_scores))
    print(project_name + " | D2H Score of init pool: " + str(d2h_scores))

    # hold best values
    ids = np.argsort(aod_scores)[::-1][:1]
    best_aod = aod_scores[ids[0]]
    best_eod = eod_scores[ids[0]]
    best_d2h = d2h_scores[ids[0]]
    best_config = evaluted_configs[ids[0]]

    # converting str value to int for CART to work
    evaluted_configs = [tuner.transform_to_numeric(x) for x in evaluted_configs]
    param_search_space = [tuner.transform_to_numeric(x) for x in param_search_space]

    # number of eval
    eval = 0
    while this_budget > 0:
        cart_model_aod = DecisionTreeRegressor(random_state=0)
        cart_model_aod.fit(evaluted_configs, aod_scores)

        cart_model_eod = DecisionTreeRegressor(random_state=0)
        cart_model_eod.fit(evaluted_configs, eod_scores)

        cart_model_d2h = DecisionTreeRegressor(random_state=0)
        cart_model_d2h.fit(evaluted_configs, d2h_scores)

        cart_models = []
        cart_models.append(cart_model_aod)
        cart_models.append(cart_model_eod)
        cart_models.append(cart_model_d2h)

        next_config_id = acquisition_fn(param_search_space, cart_models)
        next_config = param_search_space.pop(next_config_id)

        next_config_normal = tuner.reverse_transform_from_numeric(next_config)

        next_aod = measure_fitness(tuner, x_train, y_train, x_tune, y_tune, next_config_normal, test_df, biased_col, metric='aod')
        next_eod = measure_fitness(tuner, x_train, y_train, x_tune, y_tune, next_config_normal, test_df, biased_col,
                                   metric='eod')
        next_d2h = measure_fitness(tuner, x_train, y_train, x_tune, y_tune, next_config_normal, test_df, biased_col,
                                   metric='d2h')

        if np.isnan(next_aod) or next_aod == 0:
            print("NAN VAL")
            continue

        if np.isnan(next_eod) or next_eod == 0:
            print("NAN VAL")
            continue

        if np.isnan(next_eod) or next_d2h == 0:
            print("NAN VAL")
            continue

        aod_scores.append(next_aod)
        eod_scores.append(next_eod)
        d2h_scores.append(next_d2h)
        evaluted_configs.append(next_config)

        best_config = next_config_normal
        if isBetter(next_aod, best_aod) or isBetter(next_eod, best_eod) or isBetter(next_d2h, best_d2h):
            # best_config = next_config_normal
            best_aod = next_aod
            best_eod = next_eod
            best_d2h = next_d2h
            # this_budget += 1
            print(project_name + " | new AOD: " + str(best_aod) + " | new EOD: " + str(best_eod) + " | new D2H: " + str(best_d2h) + " budget " + str(this_budget))
        this_budget -= 1
        eval += 1

    print(project_name + " | Eval: " + str(eval))

    return best_config


def acquisition_fn(search_space, cart_models):
    vals = []
    predicts = []
    ids = []
    ids_only = []
    for cart_model in cart_models:
        predicted = cart_model.predict(search_space)
        predicts.append(predicted)
        ids.append(np.argsort(predicted)[::1][:1])
    for id in ids:
        val = [pred[id[0]] for pred in predicts]
        vals.append(val)
        ids_only.append(id[0])

    return bazza(ids_only, vals)

def bazza(config_ids, vals, N=20):
    dim = len(vals)
    rand_vecs = [[np.random.uniform() for i in range(dim)] for j in range(N)]
    min_val = 9999
    min_id = 0
    for config_id, val in zip(config_ids, vals):
        projection_val = 0
        for vec in rand_vecs:
            projection_val += np.dot(vec, val)
        mean = projection_val/N
        if mean < min_val:
            min_val = mean
            min_id = config_id

    return min_id


def isBetter(new, old):
    return old > new

def measure_fitness(tuner, x_train, y_train, x_tune, y_tune, configs, tune_df, biased_col, metric='aod'):
    clf = tuner.get_clf(configs)
    return get_counts(clf, x_train, y_train, x_tune, y_tune, tune_df, biased_col, metric)



def calc_f(cmat):
    # Precision
    # ---------
    prec = cmat[1, 1] / (cmat[1, 1] + cmat[0, 1])

    # Recall
    # ------
    recall = cmat[1, 1] / (cmat[1, 1] + cmat[1, 0])

    # F1 Score
    # --------
    f1 = 2 * (prec * recall) / (prec + recall)
    return f1


def filter_no_info(label, evaluated_configs, fscores):
    for i, score in enumerate(fscores):
        if np.isnan(score) or score == 0:
            del evaluated_configs[i]
            del fscores[i]

    return evaluated_configs, fscores