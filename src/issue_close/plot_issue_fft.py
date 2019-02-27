import os
import plotly
# plotly.tools.set_credentials_file(username='dichen001', api_key='czrCH0mQHmX5HLXSHBqS')
plotly.tools.set_credentials_file(username='amritbhanu', api_key='cuaXxPfbSxptk2irXf7P')
import plotly.plotly as py
import plotly.graph_objs as go
import cPickle
import pickle

cwd = os.getcwd()
data_path = os.path.join(cwd,"..","..", "data", "issue_close_time")
details_path = os.path.join(data_path, 'issue_close_time_details_5x10_mdlp_365.pkl')
details = cPickle.load(open(details_path, 'rb'))

with open(os.path.join(data_path, 'dodge.pickle'), 'rb') as handle:
    dodge = pickle.load(handle)


# folder = "1 day"
# folder = "7 days"
# folder = "14 days"
# folder = "30 days"
# older = "90 days"
# folder = "180 days"
folder = "365 days"
details = details[folder]
titles = details.keys()
titles.remove("hadoop.csv")

classifiers = ["DT", "RF", "LR", "kNN", "FFT-Dist2Heaven", "Dodge_0.2_30"]
colors = ["#AED6F1", "#5DADE2", "#2874A6", "#1B4F72", "#000000", "#FF5722"]#, "#E53935"]

l = len(details[titles[0]][classifiers[0]]['dist2heaven'])
x = []
for t1 in titles:
    x.extend([t1] * l)
data = []
x1 = []
for t1 in titles:
    x1.extend([t1]*21)

# print(details) file, learner, measure
# print(details["hive.csv"]["DT"].keys())

for i, clf in enumerate(classifiers):
    y = []
    for n1 in titles:
        if clf!="Dodge_0.2_30":
            y.extend(sorted(details[n1][clf]['dist2heaven']))
        else:
            y.extend(sorted(dodge[folder][n1]))
    if clf != "Dodge_0.2_30":
        tmp_bar = go.Box(
            y=y,
            x=x,
            name=clf,
            marker=dict(
                color=colors[i]
            )
        )
    else:
        tmp_bar = go.Box(
            y=y,
            x=x1,
            name=clf,
            marker=dict(
                color=colors[i]
            )
        )
    data.append(tmp_bar)

layout = go.Layout(
    title=folder + ' 25 times',
    yaxis=dict(
        title='Distance to Heaven',
        zeroline=False
    ),
    xaxis=dict(
        title='Issue Close Time Dataset',
        zeroline=False
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename=folder + " - 25 times")
