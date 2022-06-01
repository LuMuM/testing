from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_csv("C:/Users/bourh/OneDrive/Bureau/CoursL3/Stage/Exercice1.csv", header=0, index_col=0, sep=",", decimal=".")

# plt.plot(data, "purple")    # Créer une courbe
# plt.ylabel("Taille")
# plt.xlabel("Âge")
# plt.show()  # Afficher la courbe

plt.hist(data, color='purple') # Créer un histogramme
plt.grid(True)
plt.title("Histogramme")
plt.ylabel("Âge")
plt.xlabel("Taille")
plt.show()

# plt.scatter(data, data2) # Créer le nuage
# plt.show() #Afficher le nuage
