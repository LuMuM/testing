#on va importer pandas
import pandas as pd
# on importe scipy afin de faire les stats inférentielles
from scipy.stats import shapiro #on prend de quoi faire un test de shapiro
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import chi2_contingency,ttest_ind,f_oneway,mannwhitneyu,wilcoxon

from sklearn.feature_selection import chi2

#on va importer nos données grace à pandas
data = pd.read_csv("Abdourahmane\Exercice1.csv",header=0,sep=',',decimal='.')

# effectuons un test de shapiro sur nos données (test de normalité)
# cela permettra de voir si elles sont gaussiennes, si leur distribution suit la forme d'une courbe en cloche

print()
print()
print(shapiro(data)) # on fait un test de shapiro que l'on affiche ensuite
print()
print(shapiro(data['age']))  #
print(shapiro(data['taille'])) #

    # passons aux tests de correlation

        # test de pearson
print()
print("----Pearson")
print (pearsonr(data['age'],data['taille']))

        # test de spearmann
print()
print("----Spearman")
print(spearmanr(data))



        #Test du chi 2
print()
print("----chi 2")
print(chi2_contingency(data))

print()
print(chi2_contingency(data['age']))
print(chi2_contingency(data['taille']))

    #Les test paramétriques

        #Test de Student
print()
print("----Student")
print(ttest_ind(data['age'],data['taille']))

        #Annova (test d'analyse de la variance)
print()
print("----Annova")
print(f_oneway(data['age'],data['taille']))

    #test non paramétriques
        #test de mannwhitney
print()
print("----Mann Whitney")
print(mannwhitneyu(data['age'],data['taille']))

        #Test de Wilcoxon
print()
print("----Wilcoxon")
print(wilcoxon(data['age'],data['taille']))
