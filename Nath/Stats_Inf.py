from os import stat
import pandas as pd
from pandas import read_csv
import numpy as np
from scipy import stats

    
donnees=pd.read_csv('C:\\Users\\33669\\Documents\\testing\\Nath\\Exercice1.csv', header=0,sep=',', decimal='.')
print(donnees)
print(donnees.columns)

#TEST DE NORMALITE
#Avec shapiro
from scipy.stats import shapiro
print(shapiro(donnees))
##nous avons un pvalue assez significative 
##donc les donnees ne suivent pas une loi normale


#TEST DE CORRELATION 
# (on cherche à voir comment évolue le lien antre les variables)
#Il faut plusieurs variables car il s'agit d'un test de correlation

#Avec Pearson
from scipy.stats import pearsonr
print(pearsonr(donnees["age"], donnees["taille"]))
#Avec Spearman
from scipy. stats import spearmanr
print(spearmanr(donnees["age"], donnees["taille"]))
##Pearson = mesure intervalle, Pearman = ordinale 
##Pearson est plus adaptée ici

#Avec Chi-Deux
from scipy.stats import chi2_contingency 
print(chi2_contingency(donnees["age"], donnees["taille"]))

#Les trois tests nous montrent qu'il y a un poucenetage élevé de correlation

#TEST PARAMETRIQUES
#TAvec student
from scipy.stats import ttest_ind 
print(ttest_ind(donnees["age"],donnees["taille"]))

#Avec ANOVA (test de la variance)
from scipy.stats import f_oneway
print(f_oneway(donnees["age"],donnees["taille"]))

#TEST NON PARAMETRIQUE
#Avec Mann-Whithney
from scipy.stats import mannwhitneyu
print(mannwhitneyu(donnees["age"],donnees["taille"]))

