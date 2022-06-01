from token import EXACT_TOKEN_TYPES
import pandas as pd 
from pandas import read_csv 

data=pd.read_csv('C:\\Users\\Lama\MonProjet\\Exercice1.csv', header=0 ,sep=',' , decimal='.')
print(data)


#----test de normalité--------#
from scipy.stats import shapiro
print(shapiro(data)) 
#test de normalité: pour savoir si ils sont normalisées
#on trouve p-value tres petite donc significative et une correlation tres forte 




#--------test de correlation---#
from scipy.stats import pearsonr 
print(pearsonr(data.taille, data.age)) # correlation forte et p-value tres petite
#la difference entre: spearman et pearson: test de correlation
#pearson: les valeurs exact 
#spearman: les valeurs ordonnés

from scipy.stats import spearmanr #on trouve correlation tres forte et p-value tres petite
print(spearmanr(data)) 

from scipy.stats import chi2_contingency 
print(chi2_contingency(data)) 


#---test parametriques-------#
from scipy.stats import ttest_ind # Librairie
print(ttest_ind(data.taille, data.age))
#student: comparer les populations : dans in ic 

from scipy.stats import f_oneway # Librairie
stat, p = f_oneway(data.taille, data.age) 
#Anova: comparer la variances de toutes les variables et savoir si ils sont liés ou pas 


from scipy.stats import mannwhitneyu # Librairie
stat, p = mannwhitneyu(data.taille, data.age)# Fonction où data1 et data2 représentent les jeux de données à comparer
#mann-whitny: petit ech mais comparer les pop en valeurs


from scipy.stats import wilcoxon # Librairie
print(wilcoxon(data.taille, data.age) )# Fonction où data1 et data2 représentent les jeux de données à comparer
#wilcoxon: petit ech mais comparer les rangs

