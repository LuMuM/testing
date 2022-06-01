
import pandas as pd
data = pd.read_csv("Li/Exercice1.csv",header=0,sep= ',', decimal='.')

## Tests paramétriques
# Test de Student
from scipy.stats import ttest_ind 
print("Test de Student: ",ttest_ind(data['age'], data['taille']))

# Test d’Analyse de la Variance (ANOVA)
from scipy.stats import f_oneway
stat, ano=f_oneway(data['age'], data['taille'])
print("Test ANOVA: ",ano)

## Tests non-paramétriques
# Test Mann-Whitney
from scipy.stats import mannwhitneyu
stat, man = mannwhitneyu(data['age'], data['taille']) 
print ("Test Mann-Whitney: ",man)

# Test de Wilcoxon
from scipy.stats import wilcoxon
print("Test de Wilcoxon: ",wilcoxon(data['age'], data['taille']))