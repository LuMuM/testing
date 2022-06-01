from matplotlib import pyplot as plp
import math
from pandas import read_csv
import pandas as pd
import numpy as np
from scipy import stats


data = pd.read_csv('C:\\Users\\deche\\Documents\\GitHub\\testing\\Luan\\Exercice1.csv', header = 0, sep=',', decimal='.')
print(data)

#Dimensions du df
print(data.shape)

#Colonne age
print(data['age'])

print(data.columns)

plp.plot(data)
plp.show()

plp.hist(data)
plp.show()

plp.scatter(data['age'],data['taille'])
plp.show()

print(data.describe())

print(np.sum(data['taille'])) #Somme taille
print(np.mean(data['age']))   #Moyenne age
print(np.median(data['age'])) #Médiane age
var_taille = np.var(data['taille']) #Variance taille
print(var_taille)
print(np.std(data['taille'])) #Écart-type taille
print(math.sqrt(var_taille))   
print(np.percentile(data['age'],10))


print(stats.mode(data))

