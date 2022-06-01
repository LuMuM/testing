from turtle import header 
import matplotlib.pyplot as plt
import pandas as pd 
from pandas import read_csv
import random

#importation des donnée
data = pd.red_csv('C:\\Users\\Yvan\\Documents\\python\\Exercice1(1).csv', header=0, sep=',', decimal ='.')
print(data)

plt.plot(data,color ='g',linewidth =25) #Afficher le graphique
plt.title ('Graphique') #titre 
plt.xlabel ('taille') 
plt.ylabel ('age')
plt.grid(True) #Afficher la grille ou non

plt.show()