import numpy
import pandas as pd 
from pandas import read_csv
import random
from scipy.stats import shapiro # Librairie
from scipy.stats import pearsonr # Librairie
from scipy.stats import spearmanr # Librairie
from scipy.stats import chi2_contingency # Librairie

#importation des donnée
data = pd.red_csv('C:\\Users\\Yvan\\Documents\\python\\Exercice1(1).csv', header=0, sep=',', decimal ='.')
print(data)

#Test de Shapiro-Wilk
shapiro(data) # Fonction (où data représente le tableau de données, qu'on peut nommer dinosare ou pancakes si on veut
print(shapiro)

#Test de Pearson
pearsonr(data) # Fonction
print(pearsonr)

#Test de Spearman
spearmanr(data) # Fonction
print(spearmanr)

#Test du Chi-Deux
chi2_contingency(data) # Fonction
print(chi2_contingency)
