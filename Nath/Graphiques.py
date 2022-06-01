import pandas as pd
from pandas import read_csv

    
data=pd.read_csv('C:\\Users\\33669\\Documents\\testing\\Nath\\Exercice1.csv', header=0,sep=',', decimal='.')
print(data)
print(data.columns)

#Afficher le nb de ln et de cln
print(data.shape) 

#On commence par installer matplotlib
#On importe pyplot
from matplotlib import pyplot
import matplotlib.pyplot as plt


#Afficher la courbe pour le jeu de données
plt.plot(data)
plt.show()

#Afficher l'histogramme du jeu de données
plt.hist(data)
plt.show()

#Afficher le nuage de points du jeu de données
plt.scatter(data["age"], data["taille"])
plt.show()

#Afficher un camembert des ages 
plt.pie(data["age"])
plt.show()
