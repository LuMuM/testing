import imp
from matplotlib import pyplot 

# Courbe
import pandas as pd
data = pd.read_csv("Li/Exercice1.csv",header=0,index_col=0,sep= ',', decimal='.')
pyplot.plot(data) 
pyplot.show() 

# Histogramme
from matplotlib import pyplot 
import matplotlib.pyplot as plt 

plt.hist(data) 
plt.show()

# Nuage
from matplotlib import pyplot 

plt.title("age et taille", fontsize=24)
plt.xlabel("age", fontsize=14)
plt.ylabel("taille", fontsize=14)
plt.scatter(data,data) 
plt.show() 