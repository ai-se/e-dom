from __future__ import print_function, division

__author__ = 'amrit'

import matplotlib.pyplot as plt
import os
import pickle
import plotly
plotly.tools.set_credentials_file(username='amritbhanu', api_key='9S1jgWyw5vNhtZ3UlVHh')
import plotly.plotly as py
import numpy as np
from collections import OrderedDict
from operator import itemgetter

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
        li=dic[i].values()
        li=[y+(0.01*(x+1)) for y in li]
        ## li = [y - (0.01 * (x + 1)) for y in li]
        plt.plot(li,color=colors[x],label=str(i)+" epsi")

    plt.ylabel("Max Popt20 Score")
    plt.ylim(0,1.2)

    plt.xlabel("No. of iterations")
    plt.legend(bbox_to_anchor=(0.7, 0.5), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../results/popt20/"+f+ ".png")
    plt.close(fig)

def draw_iqr(dic,f):
    font = {'size': 70}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 20, 'legend.fontsize': 70, 'axes.labelsize': 80, 'legend.frameon': True,
                  'figure.autolayout': True,'axes.linewidth':8}
    plt.rcParams.update(paras)
    colors = ['red', 'green', 'blue', 'orange']
    markers=["o","*","v","D"]
    fig = plt.figure(figsize=(80, 60))
    for x,i in enumerate(e_value):
        li = dic[i].values()
        med = [round(np.median(y),3) for y in li]
        iqr = [round((np.percentile(y,75)-np.percentile(y,25)), 3) for y in li]
        plt.plot(med,color=colors[x],label="median "+str(i)+" epsi")
        plt.plot(iqr, color=colors[x],linestyle='-.', label="iqr "+str(i) + " epsi")

    plt.ylabel("Max Popt20 Score")
    plt.ylim(0,1)
    plt.xlabel("No. of iterations")
    plt.legend(bbox_to_anchor=(0.7, 0.5), loc=1, ncol=1, borderaxespad=0.)
    plt.savefig("../results/popt20/"+f+ "_iqr.png")
    plt.close(fig)

def draw_boxplot(dic,f):
    font = {'size': 70}
    plt.rc('font', **font)
    paras = {'lines.linewidth': 70, 'legend.fontsize': 70, 'axes.labelsize': 80, 'legend.frameon': True,
             'figure.autolayout': True, 'axes.linewidth': 8}
    plt.rcParams.update(paras)

    boxprops = dict(linewidth=9, color='black')
    colors = ['red', 'green', 'blue', 'purple']
    whiskerprops = dict(linewidth=5)
    medianprops = dict(linewidth=8, color='firebrick')

    dic1 = OrderedDict(sorted(dic.items(), key=itemgetter(0)))

    fig1, ax1 = plt.subplots(figsize=(80, 60))
    bplot = ax1.boxplot(dic1.values(), showmeans=False, showfliers=False, medianprops=medianprops, capprops=whiskerprops,
                       flierprops=whiskerprops, boxprops=boxprops, whiskerprops=whiskerprops)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set(color=color)
    ax1.set_xticklabels(dic1.keys())
    #ax1.set_ylim([0, 1])
    ax1.set_xlabel("Epsilon Values")
    ax1.set_ylabel("AUC of Popt20 (20 repeats)", labelpad=30)
    plt.savefig("../results/popt20/" + f + "_auc.png")
    plt.close(fig1)


if __name__ == '__main__':
    temp_fi={}
    for i in files:
        print(i)
        dic=dump_files(i)
        print(dic["settings"])
        # draw(dic['temp'],i)
        # draw_iqr(dic['counter_full'], i)
        # del dic["temp"]
        # del dic["time"]
        # del dic["counter_full"]
        # del dic["settings"]
        #draw_boxplot(dic,i)

        break

