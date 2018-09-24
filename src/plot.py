from __future__ import print_function, division

__author__ = 'amrit'

import matplotlib.pyplot as plt
import os
import pickle
from collections import OrderedDict
from operator import itemgetter
from sklearn.metrics import auc

e_value=[0.2,0.1, 0.05,0.025]
files=["ivy","camel","jedit","log4j","lucene","poi","synapse","velocity","xalan","xerces"]
#files=["ivy","log4j","synapse","velocity", "ant","arc","camel","poi","prop","velocity","jedit"
#       ,"log4j","redaktor","tomcat","xalan","xerces"]

ROOT=os.getcwd()

def dump_files(f=''):
    # for _, _, files in os.walk(ROOT + "/../dump/defect/"):
    #     for file in files:
    #         if f in file:
    with open("../dump/defect/popt20_" + f+".pickle", 'rb') as handle:
        final = pickle.load(handle)
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
    for x,i in enumerate(e_value):
        li=dic[i]
        li=[y+(0.01*(x+1)) for y in li]
        plt.plot(li,color=colors[x],label=str(i)+" epsi")

    plt.ylabel("Max popt Score")
    plt.ylim(0,1)
    plt.xlabel("No. of iterations")
    plt.legend(bbox_to_anchor=(0.7, 0.5), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../results/popt/"+f+ ".png")
    plt.close(fig)

def draw_boxplot(dic,f):
    pass

if __name__ == '__main__':
    temp_fi={}
    for i in files:
        print(i)
        dic=dump_files(i)
        ## draw the graph
        #draw(dic['temp'],i)
        del dic["temp"]
        del dic["time"]
        draw_boxplot(dic,i)
        break
