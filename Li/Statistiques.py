
## Statistiques descriptives

from matplotlib.pyplot import show
import numpy
import pandas as pd
data = pd.read_csv("Li/Exercice1.csv",header=0,sep= ',', decimal='.')

df = pd.DataFrame(data)
df.describe()
print(data)
print()
print(df.describe())

print("``````````````````````````````````````````````````````````````````")
# numpy.sum(data)
print("sum de age:",numpy.sum(data['age']),"sum de taille:",numpy.sum(data['taille']))
print("``````````````````````````````````````````````````````````````````")
#numpy.mean(data)
print("mean de age:",numpy.mean(data['age']),"mean de taille:",numpy.sum(data['taille']))
print("``````````````````````````````````````````````````````````````````")
#numpy.median(data)
print("median de age:",numpy.median(data['age']),"median de taille:",numpy.sum(data['taille']))
print("``````````````````````````````````````````````````````````````````")
#numpy.var(data)
print("var de age:",numpy.var(data['age']),"var de taille:",numpy.sum(data['taille']))
print("``````````````````````````````````````````````````````````````````")
#numpy.std(data)
print("std de age:",numpy.std(data['age']),"std de taille:",numpy.sum(data['taille']))
print("``````````````````````````````````````````````````````````````````")
#numpy.percentile(data,50)
print("percentile de age:",numpy.percentile(data['age'],50),"percentile de taille:",numpy.sum(data['taille']))
print("``````````````````````````````````````````````````````````````````")

from scipy import stats
stats.mode(data)


##  Calcus
#from sklearn import metrics
#round()
#donnees_vrai = [data['taille']]
#donnees_pred = 
#r2 = metrics.r2_score(donnees_vrai,donnees_pred)
