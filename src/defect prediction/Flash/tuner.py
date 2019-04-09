from __future__ import print_function
__author__ = 'amrit'

import random

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class DT_TUNER:
    def __init__(self, random_state=0):
        self.criterion = ['gini', 'entropy']
        self.splitter = ['best', 'random']
        self.min_samples_split = [0.0, 1.0]
        random.seed(random_state)

        encoder_criterion = LabelEncoder()
        self.encoder_criterion = encoder_criterion.fit(self.criterion)
        encoder_splitter = LabelEncoder()
        self.encoder_splitter = encoder_splitter.fit(self.splitter)

        self.default_config = ('gini', 'best', 2)

    def generate_param_combinaions(self):
        criterion = random.choice(self.criterion)
        splitter = random.choice(self.splitter)
        min_samples_split = random.uniform(self.min_samples_split[0], self.min_samples_split[1])


        return criterion, splitter, min_samples_split

    def criterion_transform(self, val):
        arr_list = self.encoder_criterion.transform([val])
        return float(arr_list.tolist()[0])

    def criterion_reverse_transform(self, val):
        arr_list = self.encoder_criterion.inverse_transform([int(val)])
        return arr_list.tolist()[0]

    def splitter_transform(self, val):
        arr_list = self.encoder_splitter.transform([val])
        return float(arr_list.tolist()[0])

    def splitter_reverse_transform(self, val):
        arr_list = self.encoder_splitter.inverse_transform([int(val)])
        return arr_list.tolist()[0]

    def generate_param_pools(self, size):
        list_of_params = [self.generate_param_combinaions() for x in range(size)]
        return list_of_params

    def get_clf(self, configs):
        clf = DecisionTreeClassifier(criterion=configs[0], splitter=configs[1], min_samples_split=configs[2])
        return clf

    def transform_to_numeric(self, x):
        return self.criterion_transform(x[0]), self.splitter_transform(x[1]), x[2]

    def reverse_transform_from_numeric(self, x):
        return self.criterion_reverse_transform(x[0]), self.splitter_reverse_transform(x[1]), x[2]


class SVM_TUNER:
    def __init__(self, random_state=0):
        self.C_VALS = [1, 50]
        self.KERNELS = ['rbf', 'linear', 'sigmoid', 'poly']
        self.GAMMAS = [0, 1]
        self.COEF0S = [0, 1]
        self.enc = None
        random.seed(random_state)

        self.label_coding()

        self.default_config = (1.0, 'rbf', 1.0, 0)

    def generate_param_combinaions(self):
        c = random.uniform(self.C_VALS[0], self.C_VALS[1])
        kernel = random.choice(self.KERNELS)
        gamma = random.uniform(self.GAMMAS[0], self.GAMMAS[1])
        coef0 = random.uniform(self.COEF0S[0], self.COEF0S[1])

        return c, kernel, gamma, coef0

    def label_coding(self):
        enc = LabelEncoder()
        enc.fit(self.KERNELS)
        self.enc = enc

    def label_transform(self, val):
        arr_list = self.enc.transform([val])
        return float(arr_list.tolist()[0])

    def label_reverse_transform(self, val):
        arr_list = self.enc.inverse_transform([int(val)])
        return arr_list.tolist()[0]

    def generate_param_pools(self, size):
        list_of_params = [self.generate_param_combinaions() for x in range(size)]
        return list_of_params

    def get_clf(self, configs):
        clf = SVC(C=configs[0], kernel=configs[1], gamma=configs[2],
                                     coef0=configs[3], random_state=0)
        return clf


    def transform_to_numeric(self, x):
        return x[0], self.label_transform(x[1]), x[2], x[3]

    def reverse_transform_from_numeric(self, x):
        return x[0], self.label_reverse_transform(x[1]), x[2], x[3]


class LR_TUNER:
    def __init__(self, random_state=0):
        self.C_VALS = [.01, 50]
        self.PENALTY  = ['l1', 'l2']
        self.SOLVER = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        self.MAX_ITER = [50, 500]

        self.enc_solver = None
        self.enc_penalty = None
        random.seed(random_state)

        self.label_coding()

        self.default_config = (1.0, 'l2', 'liblinear', 100)

    def generate_param_combinaions(self):
        found = False
        c = random.uniform(self.C_VALS[0], self.C_VALS[1])
        iteration = random.randint(self.MAX_ITER[0], self.MAX_ITER[1])
        while not found:
            penalty = random.choice(self.PENALTY)
            solver = random.choice(self.SOLVER)
            if penalty == 'l1' and solver in ['liblinear']:
                found = True
            elif penalty == 'l2':
                found = True

        return c, penalty, solver, iteration

    def label_coding(self):
        enc = LabelEncoder()
        enc.fit(self.SOLVER)
        self.enc_solver = enc

        enc = LabelEncoder()
        enc.fit(self.PENALTY)
        self.enc_penalty = enc

    def label_transform_solver(self, val):
        arr_list = self.enc_solver.transform([val])
        return float(arr_list.tolist()[0])

    def label_reverse_transform_solver(self, val):
        arr_list = self.enc_solver.inverse_transform([int(val)])
        return arr_list.tolist()[0]

    def label_transform_penalty(self, val):
        arr_list = self.enc_penalty.transform([val])
        return float(arr_list.tolist()[0])

    def label_reverse_transform_penalty(self, val):
        arr_list = self.enc_penalty.inverse_transform([int(val)])
        return arr_list.tolist()[0]

    def generate_param_pools(self, size):
        list_of_params = [self.generate_param_combinaions() for x in range(size)]
        return list_of_params

    def get_clf(self, configs):
        clf = LogisticRegression(C=configs[0], penalty=configs[1], solver=configs[2],
                                     max_iter=configs[3], random_state=0)
        return clf


    def transform_to_numeric(self, x):
        return x[0], self.label_transform_penalty(x[1]), self.label_transform_solver(x[2]), x[3]

    def reverse_transform_from_numeric(self, x):
        return x[0], self.label_reverse_transform_penalty(x[1]), self.label_reverse_transform_solver(x[2]), x[3]



