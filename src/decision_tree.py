from __future__ import print_function, division

__author__ = 'amrit'
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import *
import graphviz
from sklearn.datasets import load_iris
iris = load_iris()

if __name__ == '__main__':
    df=pd.read_csv("../results/variance.csv")
    df.drop("dataset",axis=1,inplace=True)
    dic={}
    classes=df.classifier.unique()
    for i,j in enumerate(classes):
        dic[j]=i
    df['classifier']=df['classifier'].apply(lambda x: dic[x])
    X=df[df.columns[:-1]]
    y=df[df.columns[-1]]

    clf = DecisionTreeClassifier(criterion='entropy').fit(X, y)
    dot_data = export_graphviz(clf, out_file=None, feature_names = df.columns[:-1].values.tolist()
                               , class_names=classes,impurity=False,filled=True,
                               leaves_parallel=True)
    graph = graphviz.Source(dot_data)
    graph.render("iris")

    # dot_data = export_graphviz(clf, out_file=None,
    # feature_names = iris.feature_names,
    # class_names = iris.target_names,
    # filled = True, rounded = True,
    # special_characters = True)
    # graph = graphviz.Source(dot_data)
    # graph