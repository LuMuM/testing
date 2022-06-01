#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:01:08 2022

@author: antoinesedoh
"""
import random
def jeu():
    #générer le nombre mystère
    random.seed(None) #initialiser le générateur
    mystere = random.randint(1,9)
    #Pour rendre le jeu un peu plus difficile
    #mystere = random.randint(1,100)
    #tag booléen pour recherche
    trouve = False
    #compteur d'essais
    compteur = 0
    #recherche
    while (trouve == False):
        #saisir la valeur
        valeur = int(input("Saisir votre nombre : "))
        compteur = compteur + 1
        #tester
        if (valeur == mystere):
            trouve = True
        else:
            if (valeur > mystere):
                print("le nombre mystère est plus petit")
            else:
                print("le nombre mystère est plus grand")

    #affichage
    print("BRAVO, nombre d'essais = " + str(compteur))

#*** PROGRAMME PRINCIPAL ***
jeu()
#*** FIN PROGRAMME PRINCIPAL ***
