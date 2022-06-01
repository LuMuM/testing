import pandas as pd
from pandas import read_csv

data=pd.read_csv('C:\\Users\\loubn\\Desktop\\tutu\Exercice1.csv', header=0, sep=',', decimal='.')#importer la base
print(data)
print(data.describe())


#----------------------------------
from matplotlib import pyplot as ply
ply.plot(data, color='red', marker='o', markersize=5)
ply.ylabel('taille')
ply.xlabel('age')
ply.title('courbe ages et tailles')
ply.grid(True)
ply.show()
#----------------------------------
import numpy # Librairie
numpy.sum(['age'])
numpy.mean(['age']) # Moyenne
numpy.median(['age']) # Médiane
numpy.var(['age']) # Variance
numpy.std(['age']) # Ecart-Type
numpy.percentile(['age'], 25) 
#-----------------------------------------
numpy.sum(['taille'])
numpy.mean(['taille']) # Moyenne
numpy.median(['taille']) # Médiane
numpy.var(['taille']) # Variance
numpy.std(['taille']) # Ecart-Type
numpy.percentile(['taille'], 25) 
#--------------------------------------------- 
from scipy import stats # Librairie
stats.mode(data) #variable avec le plus gros effectif