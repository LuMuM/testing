#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:26:21 2022

@author: antoinesedoh
"""

# Fonction qui remplace les éléments en dehors de ceux présents sur la diagonale
def remplacement_pas_diag (tableau):
    # On parcourt le tableau
    for i in range(len(tableau)):
        for j in range (len(tableau)):
            # On teste si c'est bien la diagonale en vérifiant que le numéro de ligne 
            # est égal au numéro de colonne
            if (i!=j):
                tableau[i][j] = "pas diag"
    return tableau


tab = (([[1,2,3],[4,5,6],[7,8,9]]))
print("-------- Affichage 2 --------")
remplacement_pas_diag(tab)
print(remplacement_pas_diag(tab))
