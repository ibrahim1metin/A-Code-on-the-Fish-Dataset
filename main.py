import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
data=pd.read_csv("Fish.csv")
data=data.sample(frac=1).reset_index(drop=True)
y=data[["Species"]]
x=data[["Weight","Length1","Length2","Length3","Height","Width"]]
sc=StandardScaler()
x=sc.fit_transform(x)
xt1,xt2,yt1,yt2=train_test_split(x, y, test_size=0.33)
ran=RandomForestClassifier()
ran.fit(xt1,yt1)
preds=ran.predict(xt2)
acc=accuracy_score(yt2, preds)
conf=confusion_matrix(yt2, preds)
datas={i:0 for i in data.Species.unique()}
for i in data.Species:
    datas[i]+=1
plt.bar(datas.keys(), datas.values())
plt.show()
MLP=MLPClassifier(64,batch_size=20,learning_rate_init=0.001)
MLP.n_layers_=5
MLP.fit(xt1,yt1)
predsMLP=MLP.predict(xt2)
accMLP=accuracy_score(yt2, predsMLP)
confMLP=confusion_matrix(yt2,predsMLP)
print(acc)
print(accMLP)
print(conf)
print(confMLP)


searcher=GridSearchCV(ran, {
    "n_estimators":list(range(1,1001)),
    },scoring="accuracy")
searcher.fit(xt1,yt1)
print(searcher.best_score_)
print(searcher.best_params_)