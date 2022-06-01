import numpy
from scipy import stats
import pandas as pd
from pandas import read_csv
from turtle import color
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from scipy. stats import pearsonr
import metrics
import r2
from scipy. stats import pearsonr
from scipy. stats import spearmanr
from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon

#importation des données
df = pd.read_csv('C:\\Users\\HP\\Desktop\\Exercice1.csv', header = 0, sep = ',', decimal = '.')
#Resumé des données
print(df.describe())

#Mode
print(stats.mode(df))

# Arrondir la moyenne des tailles
print(round(numpy.mean(df['taille']))) 


#Shapiro
print(shapiro(df)) 
#Pearsonr
print(pearsonr(df['age'], df['taille']) )
#Spearmanr
print(spearmanr(df) )
#Chi2
print(chi2_contingency(df))

#Test de Student
print(ttest_ind(df['age'], df['taille']))

#Test Mann-Whitney
stat, p = mannwhitneyu(df['age'], df['taille'])
print( p )

#Test de Wilcoxon
print(wilcoxon(df['age'], df['taille']))