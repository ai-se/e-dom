from __future__ import print_function
__author__ = 'amrit'

import random

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.tree import DecisionTreeRegressor

from tuner import DT_TUNER
import numpy as np
import copy
from measure import get_score

BUDGET = 30
POOL_SIZE = 10000
INIT_POOL_SIZE = 12

def tune_dt(train_df,project_name,metric='d2h'):
    tuner = DT_TUNER()
    x_train, y_train=train_df[train_df.columns[:-1]],train_df[train_df.columns[-1]]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=.2)
    for train_index, tune_index in sss.split(x_train, y_train):
        x_train_flash, x_tune_flash = x_train[x_train.index.isin(train_index.tolist())], x_train[x_train.index.isin(tune_index.tolist())]
        y_train_flash, y_tune_flash = y_train[y_train.index.isin(train_index.tolist())], y_train[y_train.index.isin(tune_index.tolist())]
        best_conf = tune_with_flash(tuner, x_train_flash, y_train_flash, x_tune_flash, y_tune_flash, project_name,metric)
    return best_conf


def tune_with_flash(tuner, x_train, y_train, x_tune, y_tune, project_name,metric='d2h'):
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    #print("DEFAULT D2H: " + str(measure_fitness(tuner, x_train, y_train, x_tune, y_tune, tuner.default_config,metric)))

    this_budget = BUDGET

    # Make initial population
    param_search_space = tuner.generate_param_pools(POOL_SIZE)

    # Evaluate initial pool
    evaluted_configs = random.sample(param_search_space, INIT_POOL_SIZE)

    #param_search_space = list(set(param_search_space) - (set(evaluted_configs)))

    d2h_scores = [measure_fitness(tuner, x_train, y_train, x_tune, y_tune, configs,  metric='d2h') for configs in evaluted_configs]

    # hold best values
    ids = np.argsort(d2h_scores)[::-1][:1]
    best_d2h = d2h_scores[ids[0]]
    best_config = evaluted_configs[ids[0]]

    # converting str value to int for CART to work
    evaluted_configs = [tuner.transform_to_numeric(x) for x in evaluted_configs]
    param_search_space = [tuner.transform_to_numeric(x) for x in param_search_space]

    # number of eval
    eval = 0
    cart_models = []
    while this_budget > 0:

        cart_model_d2h = DecisionTreeRegressor()
        cart_model_d2h.fit(evaluted_configs, d2h_scores)
        cart_models.append(cart_model_d2h)
        next_config_id = acquisition_fn(param_search_space, cart_models)
        next_config = param_search_space.pop(next_config_id)

        next_config_normal = tuner.reverse_transform_from_numeric(next_config)

        next_d2h = measure_fitness(tuner, x_train, y_train, x_tune, y_tune, next_config_normal,  metric='d2h')

        d2h_scores.append(next_d2h)
        evaluted_configs.append(next_config)

        best_config = next_config_normal
        if isBetter(next_d2h, best_d2h):
            # best_config = next_config_normal
            best_d2h = next_d2h
            # this_budget += 1
            # print(project_name + " | new D2H: " + str(best_d2h) + " budget " + str(this_budget))
        this_budget -= 1
        eval += 1

    # print(project_name + " | Eval: " + str(eval))

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

def measure_fitness(tuner, x_train, y_train, x_tune, y_tune, configs, metric='d2h'):
    clf = tuner.get_clf(configs)
    clf.fit(x_train, y_train)
    test_df=copy.deepcopy(x_tune)
    y_pred = clf.predict(x_tune)
    return get_score(metric, y_pred,y_tune,test_df)


def filter_no_info(evaluated_configs, fscores):
    for i, score in enumerate(fscores):
        if np.isnan(score) or score == 0:
            del evaluated_configs[i]
            del fscores[i]

    return evaluated_configs, fscores