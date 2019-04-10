from __future__ import print_function, division

__author__ = 'amrit'
import os
import pickle
ROOT=os.getcwd()
import numpy as np

files=["ivy","camel","jedit","log4j","lucene","poi","synapse","velocity","xalan","xerces"]

def dump_files(f=''):
    with open("../../../dump/defect/Flash/XGB/XGB/flash_" + f+".pickle", 'rb') as handle:
        final = pickle.load(handle)
    return final

if __name__ == '__main__':
    temp_fi = {}
    for i in files:
        dic = dump_files(i)
        print(i,np.median(dic[i]))
