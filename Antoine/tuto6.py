#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 10:04:26 2022

@author: antoinesedoh
"""

from pandas import read_csv
import pandas as pd

data = read_csv('pokemon.csv', header = 0, index_col = 0, sep= ',', decimal = '.')
print(data)

print(data.describe()) # Equivalent à summary() en R : résume les données

import numpy # Librairie
numpy.sum(data) # Somme
numpy.mean(data) # Moyenne
numpy.median(data['defense']) # Médiane
numpy.var(data['speed']) # Variance
numpy.std(data) # Ecart-Type
numpy.percentile(data['attack'], 95) # Percentile (ici on a précisé 95,
# ou tout autre percentile entre 1 et 100))

from scipy import stats # Librairie
print(stats.mode(data)) # Mode (variable avec le plus gros effectif)

# Test de Shapiro-Wilk
from scipy.stats import shapiro # Librairie
shap = shapiro(data['attack'])

#Test de Pearson
#Test de corrélation (lien d’évolution entre des variables).
from scipy. stats import pearsonr # Librairie
pear = pearsonr(data['attack'], data['sp_attack']) # Fonction


# Test de Spearman
# Test de corrélation (lien d’évolution entre des variables) basé sur les rangs 
# (leur classement, leur position par rapport aux autres) et non les valeurs.

from scipy. stats import spearmanr # Librairie
s =spearmanr(data['defense'], data['sp_defense']) # Fonction

## Test de Khi-deux
from scipy.stats import chi2_contingency # Librairie
c = chi2_contingency(data['hp']) # Fonction


## Test de Student
from scipy.stats import ttest_ind # Librairie
t = ttest_ind(data['hp'], data['base_egg_steps']) # Fonction où data1 et data2 représentent les jeux de données à comparer, là encore, on les nommes comme on le souhaite)


#ANOVA
from scipy.stats import f_oneway # Librairie
stat, p = f_oneway(data['hp'], data['attack'], data['defense']) 


# Test de Mann-Whitney
from scipy.stats import mannwhitneyu # Librairie
stat, p1 = mannwhitneyu(data['attack'], data['speed'])

#Test de Wilcoxon
from scipy.stats import wilcoxon # Librairie
wil = wilcoxon(data['speed'], data['weight_kg'])









