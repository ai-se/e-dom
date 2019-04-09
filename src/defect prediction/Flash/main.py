from __future__ import print_function, division

__author__ = 'amrit'

import os
cwd = os.getcwd()
import_path=os.path.abspath(os.path.join(cwd, '..'))
import sys
sys.path.append(import_path)
import pandas as pd
from demo import *

from random import seed
from flash import *
from tuner import *
import pickle

metrics=["d2h","popt","popt20"]
data_path = os.path.join(cwd, "..","..","..", "data","defect")

file_dic = {"ivy":     ["ivy-1.1.csv", "ivy-1.4.csv", "ivy-2.0.csv"],\
        "lucene":  ["lucene-2.0.csv", "lucene-2.2.csv", "lucene-2.4.csv"],\
        "poi":     ["poi-1.5.csv", "poi-2.0.csv", "poi-2.5.csv", "poi-3.0.csv"],\
        "synapse": ["synapse-1.0.csv", "synapse-1.1.csv", "synapse-1.2.csv"],\
        "velocity":["velocity-1.4.csv", "velocity-1.5.csv", "velocity-1.6.csv"], \
        "camel": ["camel-1.0.csv", "camel-1.2.csv", "camel-1.4.csv", "camel-1.6.csv"], \
        "jedit": ["jedit-3.2.csv", "jedit-4.0.csv", "jedit-4.1.csv", "jedit-4.2.csv", "jedit-4.3.csv"], \
        "log4j": ["log4j-1.0.csv", "log4j-1.1.csv", "log4j-1.2.csv"], \
        "xalan": ["xalan-2.4.csv", "xalan-2.5.csv", "xalan-2.6.csv", "xalan-2.7.csv"], \
        "xerces": ["xerces-1.2.csv", "xerces-1.3.csv", "xerces-1.4.csv"]
        }

file_inc = {"ivy": 0, "lucene": 1, "poi":  2, "synapse":3, "velocity":4, "camel": 5,"jedit": 6,
            "log4j": 7, "xalan": 8,"xerces": 9}

def _test(res=''):
    seed(1)
    np.random.seed(1)

    paths = [os.path.join(data_path, file_name) for file_name in file_dic[res]]
    train_df = pd.concat([pd.read_csv(path) for path in paths[:-1]], ignore_index=True)
    test_df = pd.read_csv(paths[-1])

    train_df, test_df = train_df.iloc[:, 3:], test_df.iloc[:, 3:]

    train_df['bug'] = train_df['bug'].apply(lambda x: 0 if x == 0 else 1)
    test_df['bug'] = test_df['bug'].apply(lambda x: 0 if x == 0 else 1)

    metric = "d2h"
    dic={}
    l=[]
    for mn in range(500+file_inc[res]*10,521+file_inc[res]*10):
        np.random.seed(mn)
        seed(mn)
        best_config=tune_dt(train_df,res,metric)
        tuner=DT_TUNER()
        x_train,y_train=train_df[train_df.columns[:-1]],train_df[train_df.columns[-1]]
        x_test,y_test=test_df[test_df.columns[:-1]],test_df[test_df.columns[-1]]
        score=measure_fitness(tuner,x_train, y_train, x_test, y_test, best_config, metric)
        l.append(score)
    dic[res]=l
    print(dic)
    with open('dump/flash_' + res + '.pickle', 'wb') as handle:
        pickle.dump(dic, handle)

if __name__ == '__main__':
    eval(cmd())