import pandas as pd
from pandas import read_csv

data=pd.read_csv('C:\\Users\\loubn\\Desktop\\tutu\Exercice1.csv', header=0, sep=',', decimal='.')#importer la base
print(data)

#Test de normalité
from scipy.stats import shapiro # Librairie
print(shapiro(data)) #p-value tres petit donc les données suivent une loi normale, rejette H0

#Test de corrélation
from scipy.stats import pearsonr # Librairie
print(pearsonr(data.age,data.taille))#on a une corrélation presque égale à 1 donc si la taille augmente l'age augemnt tres forte

from scipy. stats import spearmanr # Librairie
print(spearmanr(data) )


from scipy.stats import chi2_contingency # Librairie
print(chi2_contingency(data)) 

#Tests paramétriques
#student
from scipy.stats import ttest_ind # Librairie
print(ttest_ind(data.age, data.taille)) # p-value significative alors les moyennes de 
#deux grands échantillons indépendants sont significativement différentes.

