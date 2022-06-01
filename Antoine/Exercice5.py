#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:32:43 2022

@author: antoinesedoh
"""
#Créer une fonction qui remplace tous les termes dans la diagonale d'un tableau
# carré par la chaîne "diag". 

def remplacement_diag (tableau):
    for i in range(len(tableau)):
        for j in range (len(tableau)):
            if (i==j):
                tableau[i][j] = "diag"
    return tableau



tab = (([[0,1,2,3],[4,5,6,7],[7,8,9,3],[7,8,9,0]]))
remplacement_diag(tab)
print(remplacement_diag(tab))


