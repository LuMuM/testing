import pandas as pd 
from pandas import read_csv 
data=pd.read_csv('C:\\Users\\Lama\MonProjet\\Exercice1.csv', header=0 ,sep=',' ,index_col=0, decimal='.')
print(data)
#-------------------#

from matplotlib import pyplot as plt
plt.plot(data, color='yellow', linestyle = ':',marker = 'o', markersize = 10) # Créer une courbe
plt.ylabel('taille')
plt.xlabel('age')
plt.title('Courbe des ages et tailles')
plt.grid(True)
plt.show() #Afficher la courbe

#--------------------#

plt.hist(data, bins = 5, color = 'skyblue') # Créer un histogramme
plt.ylabel('taille')
plt.xlabel('age')
plt.title('Histogramme des ages et tailles')
plt.show() # Afficher un histogramme


#---------stat descreptives-----------#
print(data.describe())

#-------------------------#
import numpy as np
print(np.sum(["age"]))# Somme
print(np.mean(["age"])) # Moyenne
print(np.median(["age"])) # Médiane
print(np.var(["age"])) # Variance
print(np.std(["age"])) # Ecart-Type
print(np.percentile(["age"], 25))


print(np.sum(["taille"]))# Somme
print(np.mean(["taille"])) # Moyenne
print(np.median(["taille"])) # Médiane
print(np.var(["taille"])) # Variance
print(np.std(["taille"])) # Ecart-Type
print(np.percentile(["taille"], 25))


#------------stat inferentielle------#

from scipy import shapiro
shapiro(data) # Fonction (où data représente le tableau de données, qu'on peut nommer dinosare ou pancakes si on veut

from scipy import stats # Librairie
stats.mode(data) # Mode (variable avec le plus gros effectif)

from scipy import spearmanr # Librairie
spearmanr(data) # Fonction

from scipy import chi2_contingency # Librairie
chi2_contingency(data) # Fonction

