from __future__ import print_function, division

__author__ = 'amrit'

import matplotlib.pyplot as plt
import os
import pickle
from collections import OrderedDict
from operator import itemgetter

e_value=[0.025,0.05,0.1,0.2]
files=["ivy","log4j","synapse","velocity", "ant"]

ROOT=os.getcwd()

def dump_files(f=''):
    final={}
    for _, _, files in os.walk(ROOT + "/../dump/"):
        for file in files:
            if f in file:
                with open("../dump/" + file, 'rb') as handle:
                    dic = pickle.load(handle)
                    dic1=OrderedDict(sorted(dic.values()[0].items(), key=itemgetter(0)))
                    dic[dic.keys()[0]]=dic1.values()
                    final.update(dic)
    return final


def draw(dic,f):
    font = {'size': 70}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 20, 'legend.fontsize': 70, 'axes.labelsize': 80, 'legend.frameon': True,
                  'figure.autolayout': True,'axes.linewidth':8}
    plt.rcParams.update(paras)
    colors = ['red', 'green', 'blue', 'orange']
    markers=["o","*","v","D"]
    fig = plt.figure(figsize=(80, 60))
    for x,i in enumerate(dic.keys()):
        plt.plot(dic[i],color=colors[x],label=str(i)+" epsi")

    plt.ylabel("Max Auc Score")
    plt.xlabel("No. of iterations")
    plt.legend(bbox_to_anchor=(0.7, 0.5), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../results/"+f+ ".png")
    plt.close(fig)


if __name__ == '__main__':
    temp_fi={}
    for i in files:
        dic=dump_files(i)
        draw(dic,i)
