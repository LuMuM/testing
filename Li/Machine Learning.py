
from re import X
from matplotlib.pyplot import show
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Li/Exercice1.csv",sep= ',', decimal='.',header=0,)
X=np.array(data['age'])
print("x: ",X)
y=np.array(data['taille'])
print("y: ",y)

# Régressions simple
model = LinearRegression().fit(data,data['taille'])

print("score: ",model.score(data,data['taille']))
print("coef: ",model.coef_)
print("intercept: ",model.intercept_)
print("predict : ",model.predict(data))


