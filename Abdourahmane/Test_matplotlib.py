from msilib.schema import Directory
from importlib_metadata import csv
import matplotlib as mplt
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

data = pd.read_csv("Abdourahmane\Exercice1.csv",header=0,sep=',',decimal='.')
#on peut ajouter "index_col = 0" pour dire que la première colonne numérote les individus

"""
plt.plot(data)
plt.show()

plt.hist(data)
plt.show()

"""
"""
#test de graphiques
x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])
plt.plot(x,y)
plt.show()

"""


# Partie stats descriptives
# nous allons tester quelques fonctionnalités 

print()
print(data) # on regarde comment s'affichent nos données

print()
dat_fram = pd.DataFrame(data) #fait de nos données une "dataframe"

print()
print(dat_fram) # on veut voir à quoi ressemble la data frame
#La data frame est identique à data créé plus haut

print()
print(dat_fram.describe()) # là on creer une description de nos données que l'on affiche ensuite

print()
print(np.sum(data)) #on cherche ici la somme des donnée avec numpy
#cela nous affiche la somme de chaque colonne

print()
print(np.mean(data))

print() # on va tester pour n'afficher que l'age puis la taille
print(np.mean(data['age']))
print(np.mean(data['taille']))

# print(np.mean(data['age''taille'])) #ceci ne fonctionne guère

print() #les médianes
print("Mediane : ",np.median(data))

print("Mediane : ",np.median(data['age']))
print("Mediane : ",np.median(data['taille']))



print() #les variances
print(np.var(data['age']))
print(np.var(data['taille']))

print() #les écarts-types
print(np.std(data['age']))
print(np.std(data['taille']))


print() #les pourçentiles
# print(np.percentile(data)) #cette chose ne fonctionne point, il faut préciser

print(np.percentile(data,25))

print(np.percentile(data['age'],25))
print(np.percentile(data['taille'],25))


"""
#usage de scipy
print("lili")
stats.mode(data) # cette fonction de sipy nous renvoi le mode d'une distribution
stats.mode(data['age'])
stats.mode(data['taille'])

"""
