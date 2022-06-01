#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:07:50 2022

@author: antoinesedoh
"""


def multiple (lower, upper, nombre): 
    multiples_n = [x for x in range(lower,upper+1, 1) if x % n == 0]
    return multiples_n

#saisir la chaîne de caractères 
print ("--- Saisir l'intervalle : ---")
a = int(input("saisir le nombre de départ : "))
b = int(input("saisir le nombre de fin: "))

#Saisir le nombre dont on recherche les multiples
n = int (input("Saisir le nombre dont on recherche les multiples : "))



resultat = multiple(a, b, n)
print ("")
print("Le(s) multiple(s) de ", n, "est (sont) : ")
print(resultat)