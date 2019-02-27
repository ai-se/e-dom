import os
import plotly
# plotly.tools.set_credentials_file(username='dichen001', api_key='czrCH0mQHmX5HLXSHBqS')
plotly.tools.set_credentials_file(username='amritbhanu', api_key='cuaXxPfbSxptk2irXf7P')
import plotly.plotly as py
import plotly.graph_objs as go
import cPickle
import pickle

cwd = os.getcwd()
data_path = os.path.join(cwd,"..","..","data", "smell")
details_path = os.path.join(data_path, 'smell_details_38-MDLP.pkl')
details = cPickle.load(open(details_path, 'rb'))

with open(os.path.join(data_path, 'dodge.pickle'), 'rb') as handle:
    dodge = pickle.load(handle)

n1, n2, n3, n4 = "DataClass", "FeatureEnvy", "GodClass", "LongMethod"
t1, t2, t3, t4 = "DataClass", "FeatureEnvy", "GodClass", "LongMethod"

classifiers = ["DT", "RF", "LR", "kNN", "FFT-Dist2Heaven", "Dodge_0.2_30"]
colors = ["#AED6F1", "#5DADE2", "#2874A6", "#1B4F72", "#000000", "#FF5722"]#, "#E53935"]

data = []
l = len(details[n1][classifiers[0]]['dist2heaven'])
x = [t1] * l + [t2] * l + [t3] * l + [t4] * l
x1 = [t1] * 21 + [t2] * 21 + [t3] * 21 + [t4] * 21

for i, clf in enumerate(classifiers):
    if clf != "Dodge_0.2_30":
        tmp_bar = go.Box(
            y=sorted(details[n1][clf]['dist2heaven']) +
            sorted(details[n2][clf]['dist2heaven']) +
            sorted(details[n3][clf]['dist2heaven']) +
            sorted(details[n4][clf]['dist2heaven']),
            x=x,
            name=clf,
            marker=dict(
                color=colors[i]
            )
        )
    else:
        tmp_bar = go.Box(
            y=sorted(dodge[n1]) +
              sorted(dodge[n2]) +
              sorted(dodge[n3]) +
              sorted(dodge[n4]),
            x=x1,
            name=clf,
            marker=dict(
                color=colors[i]
            )
        )
    data.append(tmp_bar)

layout = go.Layout(
    title="Bad Smell - 25 Times",
    yaxis=dict(
        title='Distance to Heaven',
        zeroline=False
    ),
    xaxis=dict(
        title='Bad Smell Dataset (very small)',
        zeroline=False
    ),
    boxmode='group'
)
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename="Smell - 25 Times")
