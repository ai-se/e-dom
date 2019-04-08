from __future__ import print_function, division

__author__ = 'amrit'

import os
import pandas as pd
from sklearn.preprocessing import *
from mdlp.discretization import MDLP
import math
from random import seed
import numpy as np

cwd = os.getcwd()
data_path = os.path.join(cwd, "..", "data","defect")
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

data_path1 = os.path.join(cwd,"..", "data","smell")

bad_smell = {"dataclass":     ["DataClass.csv"],\
        "featureenvy":  ["FeatureEnvy.csv"],\
        "godclass":     ["GodClass.csv"],\
        "longmethod": ["LongMethod.csv"]
            }

directories= ["1 day", "7 days", "14 days", "30 days", "90 days", "180 days", "365 days"]

issue_close = {"camel": 0, "cloudstack": 1, "cocoon":  2, "deeplearning":3,"hadoop":4, "hive":5,"node":6
            , "ofbiz":7, "qpid":8}

UCI = {"adult": 0, "cancer": 1, "covtype":  2, "diabetic":3, "optdigits":4, "pendigits":5
            , "satellite":6, "shuttle":7, "waveform":8,"annealing":9,"audit":10,"autism":11,
            "bank":12,"bankrupt":13,"biodegrade":14,"blood-transfusion":15,"car":16,
            "cardiotocography":17,"cervical-cancer":18, "climate-sim":19,"contraceptive":20,
            "credit-approval":21,"credit-default":22,"crowdsource":23,"drug-consumption":24,
            "electric-stable":25,"gamma":26,"hand":27,"hepmass":28,"htru2":29,"image":30,
            "kddcup":31,"liver":32,"mushroom":33,"phishing":34,"sensorless-drive":35,"shop-intention":36}


def cal_entropy(temp,total):
    total = sum(temp)
    if len(temp)>1:
        if 0 in temp:
            val = (temp[0] / total) * math.log(temp[0] / total, 2)
        else:
            val=(temp[0]/total)*math.log(temp[0]/total,2) + (temp[1]/total)*math.log(temp[1]/total,2)
        return -(sum(temp)*val)/total
    else:
        return

def all_entropies(df1,total):
    mdlp = MDLP()
    x_test=mdlp.fit_transform(df1[df1.columns[:-1]].values, df1[df1.columns[-1]].values)
    entropies = []
    for x, y in enumerate(mdlp.cut_points_):
        if len(y) > 1:
            for j, k in enumerate(y):
                if j == 0:
                    temp = df1[df1[x] <= y[j]]['class'].value_counts().values.tolist()
                    if len(temp) > 1:
                        entropies.append(cal_entropy(temp,total))
                    else:
                        temp.append(0)
                        entropies.append(cal_entropy(temp,total))
                if j == len(y) - 1:
                    temp = df1[df1[x] > y[j]]['class'].value_counts().values.tolist()
                    if len(temp) > 1:
                        entropies.append(cal_entropy(temp,total))
                    else:
                        temp.append(0)
                        entropies.append(cal_entropy(temp,total))

                if j != len(y) - 1:
                    temp = df1[(df1[x] > y[j]) & (df1[x] <= y[j + 1])]['class'].value_counts().values.tolist()
                    if len(temp) > 1:
                        entropies.append(cal_entropy(temp,total))
                    else:
                        temp.append(0)
                        entropies.append(cal_entropy(temp,total))

        if len(y) == 1:
            temp = df1[df1[x] <= y[0]]['class'].value_counts().values.tolist()
            if len(temp) > 1:
                entropies.append(cal_entropy(temp, total))
            else:
                temp.append(0)
                entropies.append(cal_entropy(temp, total))
            temp = df1[df1[x] > y[0]]['class'].value_counts().values.tolist()
            if len(temp) > 1:
                entropies.append(cal_entropy(temp, total))
            else:
                temp.append(0)
                entropies.append(cal_entropy(temp, total))

        if len(y) == 0:
            temp = df1['class'].value_counts().values.tolist()
            if len(temp) > 1:
                entropies.append(cal_entropy(temp, total))
            else:
                temp.append(0)
                entropies.append(cal_entropy(temp, total))

    return sorted(entropies)


if __name__ == '__main__':
    seed(1)
    np.random.seed(1)
    cols=['dataset','10','30','50','70','90','samples','% class']
    l=[]
    for i in file_dic.keys():
        temp=[]
        paths = [os.path.join(data_path, file_name) for file_name in file_dic[i]]
        df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        df=df.iloc[:, 3:]
        df[df.columns[-1]]=df[df.columns[-1]].apply(lambda x: 0 if x == 0 else 1)
        for x in df.columns[:-1]:
            count=df[x].value_counts().values.tolist()
            if len(count)<=1:
                df.drop(x, axis=1, inplace=True)
        scaler = MinMaxScaler()
        df1 = pd.DataFrame(scaler.fit_transform(df[df.columns[:-1]].values))
        df1['class'] = df[df.columns[-1]]
        temp.append(i)
        total = df1['class'].count()
        vals=all_entropies(df1,total)
        vals = [i for i in vals if i != None]

        for x in [10,30,50,70,90]:
            temp.append(round(np.percentile(vals,x),2))

        temp.append(total)
        pos=df1[df1['class']==1]['class'].count()
        temp.append(round(round((pos/total),3)*100,1))
        l.append(temp)


    for i in bad_smell.keys():
        temp = []
        paths = [os.path.join(data_path1, file_name) for file_name in bad_smell[i]]
        df = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
        df[df.columns[-1]] = df[df.columns[-1]].apply(lambda x: 0 if x == 0 else 1)
        for x in df.columns[:-1]:
            count=df[x].value_counts().values.tolist()
            if len(count)<=1:
                df.drop(x, axis=1, inplace=True)

        scaler = MinMaxScaler()
        df1 = pd.DataFrame(scaler.fit_transform(df[df.columns[:-1]].values))
        df1['class'] = df[df.columns[-1]]

        temp.append(i)
        total = df1['class'].count()
        vals = all_entropies(df1,total)
        vals = [i for i in vals if i != None]

        for x in [10, 30, 50, 70, 90]:
            temp.append(round(np.percentile(vals, x), 2))

        temp.append(total)
        pos = df1[df1['class'] == 1]['class'].count()
        temp.append(round(round((pos / total), 3) * 100, 1))
        l.append(temp)

    for k in directories:
        for i in issue_close.keys():
            temp = []
            df = pd.read_csv("../data/issue_close_time/" + k + "/" + i + ".csv")
            df[df.columns[-1]] = df[df.columns[-1]].apply(lambda x: 0 if x == 0 else 1)
            for x in df.columns[:-1]:
                count=df[x].value_counts().values.tolist()
                if len(count)<=1:
                    df.drop(x, axis=1, inplace=True)

            scaler = MinMaxScaler()
            df1 = pd.DataFrame(scaler.fit_transform(df[df.columns[:-1]].values))
            df1['class'] = df[df.columns[-1]]

            temp.append(k+"_"+i)
            total = df1['class'].count()
            vals = all_entropies(df1,total)
            vals = [i for i in vals if i != None]

            for x in [10, 30, 50, 70, 90]:
                temp.append(round(np.percentile(vals, x), 2))

            temp.append(total)
            pos = df1[df1['class'] == 1]['class'].count()
            temp.append(round(round((pos / total), 3) * 100, 1))
            l.append(temp)

    for i in UCI.keys():
        temp = []
        df = pd.read_csv("../data/UCI/" + i + ".csv")
        df[df.columns[-1]] = df[df.columns[-1]].apply(lambda x: 0 if x == 0 else 1)
        for x in df.columns[:-1]:
            count = df[x].value_counts().values.tolist()
            if len(count) <= 1:
                df.drop(x, axis=1, inplace=True)

        scaler = MinMaxScaler()
        df1 = pd.DataFrame(scaler.fit_transform(df[df.columns[:-1]].values))
        df1['class'] = df[df.columns[-1]]
        temp.append(i)
        total = df1['class'].count()
        vals = all_entropies(df1,total)
        vals=[abs(i) for i in vals if i!=None]

        for x in [10, 30, 50, 70, 90]:
            temp.append(round(np.percentile(vals, x), 2))

        temp.append(total)
        pos = df1[df1['class'] == 1]['class'].count()
        temp.append(round(round((pos / total), 3) * 100, 1))
        l.append(temp)

    df_final=pd.DataFrame(l,columns=cols)
    df_final.to_csv("../results/variance.csv",index=False)

