import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt



data = pd.read_csv('Exercice1.csv', header=0 ,index_col=0 , sep=',' , decimal='.')
pyplot.plot(data)
pyplot.show()

plt.hist(data)
plt.show()