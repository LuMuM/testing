import pandas as pd
import numpy as np
from scipy import stats

data = pd.read_csv("C:/Users/bourh/OneDrive/Bureau/CoursL3/Stage/Exercice1.csv", header=0, sep=",", decimal=".")
# df = pd.DataFrame(data)

# print(data.describe())  # Résume les données
print("Somme :", np.sum(data))          # Somme
print("Moyenne :", np.mean(data.age))       # Moyenne
print("Mediane :", np.median(data.age))     # Médiane
print("Variance :", np.var(data.age))       # Variance
print("Ecart-type: ", np.std(data.age))     # Ecart-type
print("Percentile: ", np.percentile(data.age, 10))  # Percentile

#print(stats.mode(data))
