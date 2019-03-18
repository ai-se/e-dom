from __future__ import print_function, division

__author__ = 'amrit'

import pandas as pd
import os

cwd=os.getcwd()
data_path=os.path.join(cwd,"..","..","data", "UCI")

def preprocess_adult():
    path=os.path.join(data_path,"adult")
    df1=pd.read_csv(path+"/adult.data",header=None)
    df2=pd.read_csv(path+"/adult.test",header=None)
    df=pd.concat([df1, df2], ignore_index=True)
    cat_columns = [1, 3, 5, 6, 7, 8, 9, 13]

    df[cat_columns] = df[cat_columns].astype(str)
    #print(df[df[1].astype(str).str.contains('\?')].head())
    for i in cat_columns:
        df.drop(df[df[i].astype(str).str.contains('\?')].index, inplace=True)
    df.dropna(inplace=True)

    df[cat_columns]=df[cat_columns].astype('category')
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    df[14]=df[14].apply(lambda x: x.split(".")[0])
    df[14] = df[14].apply(lambda x: 0 if x==' <=50K' else 1)
    df.to_csv(data_path+"/adult.csv",index=False,header=False)

def preprocess_waveform():
    # waveform.data actual data, waveform+noise, have some noises as well with same class labels
    path=os.path.join(data_path,"waveform")
    df1 = pd.read_csv(path + "/waveform.data", header=None)
    df=df1[df1[21]!=2]
    df.to_csv(data_path + "/waveform.csv", index=False, header=False)

def preprocess_shuttle():
    path = os.path.join(data_path, "statlog_shuttle")
    df1 = pd.read_csv(path + "/shuttle.trn", header=None,sep=" ")
    df2 = pd.read_csv(path + "/shuttle.tst", header=None,sep=" ")
    df = pd.concat([df1, df2], ignore_index=True)
    df=df[df[9].isin([1,4])]
    df[9]=df[9].apply(lambda x: 1 if x==4 else 0)
    df.to_csv(data_path + "/shuttle.csv", index=False, header=False)

def preprocess_satellite():
    path = os.path.join(data_path, "statlog_satimage")
    df1 = pd.read_csv(path + "/sat.trn", header=None, sep=" ")
    df2 = pd.read_csv(path + "/sat.tst", header=None, sep=" ")
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df[36].isin([1, 4])]
    df[36] = df[36].apply(lambda x: 1 if x == 4 else 0)
    df.to_csv(data_path + "/satellite.csv", index=False, header=False)

def preprocess_pendigits():
    path = os.path.join(data_path, "pendigits")
    df1 = pd.read_csv(path + "/pendigits.tra", header=None)
    df2 = pd.read_csv(path + "/pendigits.tes", header=None)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df[16].isin([2, 4])]
    df[16] = df[16].apply(lambda x: 1 if x == 4 else 0)
    df.to_csv(data_path + "/pendigits.csv", index=False, header=False)

def preprocess_optdigits():
    path = os.path.join(data_path, "optdigits")
    df1 = pd.read_csv(path + "/optdigits.tra", header=None)
    df2 = pd.read_csv(path + "/optdigits.tes", header=None)
    df = pd.concat([df1, df2], ignore_index=True)
    df = df[df[64].isin([1, 3])]
    df[64] = df[64].apply(lambda x: 1 if x == 3 else 0)
    df.to_csv(data_path + "/optdigits.csv", index=False, header=False)

def preprocess_covtype():
    path = os.path.join(data_path, "covtype")
    df = pd.read_csv(path + "/covtype.data", header=None)
    df = df[df[54].isin([5, 4])]
    df[54] = df[54].apply(lambda x: 1 if x == 4 else 0)
    df.to_csv(data_path + "/covtype.csv", index=False, header=False)

def preprocess_cancer():
    path = os.path.join(data_path, "breast-cancer-wisconsin")
    df1 = pd.read_csv(path + "/wdbc.data", header=None)

    df1["class"]=df1[1].apply(lambda x: 1 if "M" in x else 0)
    df1.drop(labels=[0,1], axis=1, inplace=True)
    df1.to_csv(data_path + "/cancer.csv", index=False, header=False)

def preprocess_diabetic():
    path = os.path.join(data_path, "Diabetic")
    df1 = pd.read_csv(path + "/diabetic.csv")
    df1.to_csv(data_path + "/diabetic.csv", index=False, header=False)

def testing():
    df=pd.read_csv(data_path+"/diabetic.csv",header=None)
    neg = df[df[df.columns[-1]] == 0]
    pos = df[df[df.columns[-1]] == 1]
    cut_pos = int(pos[0].count() * 0.8)
    cut_neg = int(neg[0].count() * 0.8)
    pos_1, pos_2 = pos.iloc[:cut_pos, :], pos.iloc[cut_pos:, :]
    neg_1, neg_2 = neg.iloc[:cut_neg, :], neg.iloc[cut_neg:, :]
    df = pd.concat([pos_1, neg_1, pos_2, neg_2], ignore_index=True)

if __name__ == '__main__':

    # preprocess_adult()
    # preprocess_waveform()
    # preprocess_shuttle()
    # preprocess_satellite()
    # preprocess_pendigits()
    # preprocess_optdigits()
    # preprocess_covtype()
    # preprocess_cancer()
    # preprocess_diabetic()
    testing()

    # print(df.info())
    # print(df[16].value_counts())


