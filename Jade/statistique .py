import pandas as pd # Librairie


#importation des données
data = pd.read_csv('C:\\Users\\Yvan\\Documents\\python\\Exercice1(1).csv', header=0, sep=',', decimal ='.')
import numpy # Librairie

df = pd.DataFrame(data)
df.describe() # Equivalent à summary() en R : résume les données
print(df.describe)

#Pour les âges
numpy.sum(data['age']) # Somme
print(numpy.sum)

numpy.mean(data['age']) # Moyenne
print(numpy.mean)

numpy.median(data['age']) # Médiane
print(numpy.median)

numpy.var(data['age']) # Variance
print(numpy.var)

numpy.std(data['age']) # Ecart-Type
print(numpy.std)

#Pour les tailles 
numpy.sum(data['taille']) # Somme
print(numpy.sum)

numpy.mean(data['taille']) # Moyenne
print(numpy.mean)

numpy.median(data['taille']) # Médiane
print(numpy.median)

numpy.var(data['taille']) # Variance
print(numpy.var)

numpy.std(data['taille']) # Ecart-Type
print(numpy.std)

from scipy import stats # Librairie
stats.mode(data) # Mode (variable avec le plus gros effectif)

plt.plot(data) #courbe
plt.show()
