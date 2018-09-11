from __future__ import print_function, division

__author__ = 'amrit'

from transformation import *
from random import seed
import numpy as np
from utilities import _randchoice, unpack
from ML import *
from itertools import product
from sklearn.model_selection import StratifiedKFold
from demos import *
import pickle

preprocess=[standard_scaler,minmax_scaler, maxabs_scaler,[robust_scaler]*20,kernel_centerer,[quantile_transform]*200
            ,normalizer,[binarize]*100,[polynomial]*5]
MLs=[NB,[KNN]*20,[SVM]*100,[RF]*50,[DT]*30,[LR]*50]
preprocess_list=unpack(preprocess)
MLs_list=unpack(MLs)
combine=[[r[0],r[1]] for r in product(preprocess_list,MLs_list)]

func_str_dic={}
func_str_counter_dic={}
dic_value=[]
dic_median={}
final={}
e_value=[0.025,0.05,0.1,0.2]

def readfile(path=""):
    df=pd.read_csv(path)
    return df

def _test(res=''):
    np.random.seed(500)
    seed(500)
    df=readfile("../data/defect/arc.csv")
    e_val=[]
    e_val.append(float(res))

    for i in combine:
        scaler, tmp1 = i[0]()
        model, tmp2= i[1]()
        string1=tmp1+"|"+tmp2
        func_str_dic[string1]=[scaler,model]
        func_str_counter_dic[string1]=0

    for e in e_val:
        for x in range(100):
            # func_str_counter_dic=OrderedDict(sorted(func_str_counter_dic.items(), key=itemgetter(1)))
            keys=[k for k, v in func_str_counter_dic.items() if v == 0]
            key=_randchoice(keys)

            scaler,model=func_str_dic[key]
            df1=transform(df,scaler)

            skf = StratifiedKFold(n_splits=5)
            data, labels= df1[df1.columns[:-1]].values, df1[df1.columns[-1]].values
            for train_index, test_index in skf.split(data, labels):
                train_data, train_labels, test_data, test_labels=data[train_index], labels[train_index],data[test_index], labels[test_index]
                measurement=run_model(train_data,train_labels,test_data,test_labels,model)
                if measurement>0.5 and all(abs(t-measurement)>e for t in dic_value):
                    dic_value.append(measurement)
                    func_str_counter_dic[key] += 1
                else:
                    func_str_counter_dic[key] += -1
                dic_median[x]=max(dic_value)
                print(dic_value)

                break
        final[e]=dic_median
    print(final)
    with open('../dump/temp_' + str(res) + '.pickle', 'wb') as handle:
        pickle.dump(final, handle)

if __name__ == '__main__':
    eval(cmd())