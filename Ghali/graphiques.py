from turtle import color
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import random


data = pd.read_csv('C:\\Users\\HP\\Desktop\\Exercice1.csv', header = 0, index_col = 0, sep = ',', decimal = '.')

plt.plot(data,color = 'r', linewidth=25)
plt.title("Priorité de Ghali ")
plt.xlabel('Faim (en burgers)')
plt.ylabel('Temps de stage (en secondes)')
plt.grid(True)
plt.text(8, 125, r'Danger')
plt.annotate('Début de la perte d attention', xy=(8, 125), xytext=(10, 120), 
arrowprops={'facecolor':'orange', 'shrink':0.05} )


plt.show()
