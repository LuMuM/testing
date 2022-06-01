import pandas as pd
from scipy.stats import shapiro, pearsonr, spearmanr, chi2_contingency, ttest_ind, f_oneway, wilcoxon

data = pd.read_csv("C:/Users/bourh/OneDrive/Bureau/CoursL3/Stage/Exercice1.csv", header=0, sep=",", decimal=".")

# --------- Test Shapiro pour tester la normalité --------- #
print(shapiro(data))

# --------- Tests de corrélation ------------- #
# Test de Person (lien d’évolution entre des variables) (c'est mieux)
print(pearsonr(data.age, data.taille))
# Test de Spearman basé sur les rangs et non les valeurs (bien pour les notes)
print(spearmanr(data.age, data.taille))
# Si des variables qualitatives sont dépendantes.
print(chi2_contingency(data1, data2))

# -------- Tests paramétriques -------------- #
# Test de Student (si 2 grands échantillons sont liés)
ttest_ind(data1, data2)
# Test ANOVA (si plusieurs grands échantillons sont liés)

# --------- Test non-paramétriques ---------- #
# Test Mann-Whitney (Comparer distribution de 2 petits échantillons indépendants)
f_oneway(data1, data2)
# Test de Wilcoxon (Comparer distribution de 2 petits échantillons indépendants en se basant sur leurs rangs)
wilcoxon(data1, data2)
