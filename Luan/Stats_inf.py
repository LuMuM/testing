from matplotlib import pyplot as plt
from pandas import read_csv
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import shapiro,pearsonr,spearmanr,mannwhitneyu
import metrics
import sklearn
from sklearn.linear_model import LinearRegression


data = pd.read_csv('C:\\Users\\deche\\Documents\\GitHub\\testing\\Luan\\Exercice1.csv', header = 0, sep=',', decimal='.')
print(data)

#Test de Shapiro
print(shapiro(data)) #stat=0,7618, pval = 0,0002

#Test de Pearson
print(pearsonr(data['age'],data['taille'])) #0.9948, 3.034e-09

#Test de Spearman
print(spearmanr(data)) #correlation=0.99, pvalue=6.64e-64

#Test de Student
#print(ttest_ind())


p = mannwhitneyu(data['age'], data['taille'])
print(p)





