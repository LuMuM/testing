
import pandas as pd
data = pd.read_csv("Li/Exercice1.csv",header=0,sep= ',', decimal='.')

## Tests de normalité
# Test de Shapiro-Wilk
print()
from scipy.stats import shapiro
print("shapiro avec age: ",shapiro(data['age']),"shapiro avec taille: ",shapiro(data['taille']))
print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")
## Tests de corrélation
# Test de Pearson
from scipy.stats import pearsonr
import numpy as np

x=np.array(data['age'])
y=np.array(data['taille'])

ps = pearsonr(x,y)
print("Test de Pearson: ",ps)
print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")

# Test de Spearman
from scipy. stats import spearmanr
print(spearmanr(data))
print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")

# Test du Chi-Deux
from scipy.stats import chi2_contingency
print("Chi-Deux: "  ,chi2_contingency(data))
print("'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''")