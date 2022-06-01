import torch
from pandas import DataFrame as df
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
from sklearn.linear_model import LogisticRegression as logr
from matplotlib import pyplot as plt


data = pd.read_csv("Lucas/Exercice1.csv", header=0, sep=",", decimal=".")

X = np.array(data['age']).reshape((-1,1))
y = np.array(data['taille'])

for i in y :
    i.astype(float)
    
model = lr().fit(X,y)
B1 = model.coef_
B0 = model.intercept_
print('R^2 : ', model.score(X, y))

XPredict = np.array([20,30,40,29]).reshape((-1,1))

XPredict = model.predict(XPredict)
print(XPredict)


plt.scatter(X, y, s = 10)
plt.plot(X, B1*X+B0, '--b')
plt.xlabel('Age', family='serif',color='b',weight='normal', size = 16,labelpad = 6)
plt.ylabel('Taille', family='serif',color='r',weight='normal', size = 16,labelpad = 6)
plt.title("Tailles en fonction de l'age", fontdict={'family': 'serif', 'color' : 'darkblue','weight': 'bold','size': 18})
plt.grid(True) #affiche la grille


#Regression logistique

lmodel = logr().fit(X, y)
print('R^2 : ', lmodel.score(X, y))
b1 = lmodel.coef_
b0 = lmodel.intercept_


plt.show()  #affiche le graphe

