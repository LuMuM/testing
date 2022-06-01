#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:45:55 2022

@author: antoinesedoh
"""

import matplotlib.pyplot as plt # Librairie
# import matplotlib.pyplot as plt
from matplotlib import pyplot # Librairie

#librairie numpy
import numpy

#création d'un vecteur de valeurs
y = numpy.array([10,25,36,7,45,6])
x = numpy.array([1,25,12,8,15,9])
print(y)

pyplot.plot(y) # Créer une courbe
pyplot.show() #Afficher la courbe

plt.scatter(x, y) # Créer le nuage
plt.show() #Afficher le nuage


#un barplot
plt.bar(range(1,y.size+1),y)
plt.show()

#avec des couleurs différentes selon pair/impair
couleurs = numpy.where(y % 2 ==0,'aqua','lime')
print(couleurs)

#barplot de nouveau -- attention plot.show() si lancement programme
plt.bar(range(1,y.size+1),y,color=couleurs)
plt.show()

#ligne - courbe
plt.plot(y)
plt.show()


#ligne en fixant l'abscisse
plt.plot(range(2,14,2),y)

#nuage de points avec axes et titres
plt.scatter(x, y)
plt.xlabel('Abscisse')
plt.ylabel('Ordonnée')
plt.title('Titre')
plt.show()


#nuage de points avec des couleurs
plt.scatter(x, y, c=couleurs)


#autre approche pour les nuages de points
#nuage de points avec des couleurs et motifs
plt.plot(x[y%2==0], y[y%2==0], "bo")
plt.plot(x[y%2!=0], y[y%2!=0], "gD")
plt.show()



id = numpy.array([1,2,1,3,1,2])
idCouleur = numpy.array(['bo','g+','rD'])
#enchaîner les graphiques avec des couleurs différentes
for g in numpy.unique(id):
    plt.plot(x[id==g],y[id==g],idCouleur[g-1])
plt.show() #affiche le graphique

#générer des valeurs aléatoires
import scipy.stats as stat
rnd1 = stat.norm.rvs(loc=0,scale=1,size=100)
rnd2 = stat.norm.rvs(loc=1.5,scale=1.5,size=100)
rnd = numpy.concatenate([rnd1,rnd2])
print(rnd.shape)

#histogramme
plt.hist(rnd)

#deux graphiques séparés
fig,axs = plt.subplots(2)
axs[0].hist(rnd1,color='skyblue')
axs[1].hist(rnd2,color='aqua')
plt.show()

#grille pour deux graphiques placés côte-à-côte
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(1, 2)
#premier histogramme
ax = plt.subplot(gs[0,0])
ax.hist(rnd1,color='skyblue')
ax.set_title('Premier')
#second histogramme
ax = plt.subplot(gs[0,1])
ax.hist(rnd2,color='aqua')
ax.set_title('Second')
plt.show()

#2 histogrammes dans le même graphique
plt.hist([rnd1,rnd2])


