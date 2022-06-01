#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:58:35 2022

@author: antoinesedoh
"""

def factorielle (n): 
    #si on met en entier, le débordement de capacité
    #arrive très vite, d'où l'idée du flottant
    #valeur de départ
    f = 1
    #boucle for
    for i in range(1,n+1):
        f = f * i
    return f

#valeurs d'entrée
n = int(input("n : "))

#affichage
#ou bien >> 
res = factorielle(n)
print("---------- Affichage ---------")
print("Nombre : ", n)
print("Factorielle : " + str(res))

#Pour tester la récursivité
res1 = (factorielle(res))
print("---------- Affichage ---------")
print("Nombre : ", res)
print("Factorielle : " + str(res1))

