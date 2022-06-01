#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:33:31 2022

@author: antoinesedoh
"""


# Écrire une fonction dans un module Python qui, à tout chiffre/nombre compris entre 1 et 10, 
# lui appliquera un exposant que l'on spécifiera. Par exemple, si l'on souhaite 10 exposant 3, 
# nous obtenons 1000.
def calcul_exposant(nombre , exposant) :
    print ("")
    print ("--- Affichage ---")
    print ("Nombre :", nombre)
    print ("Exposant :", exposant)
    
    # On effectue le calcul 
    #for i in range(0,exposant+1):
    resultat = nombre ** exposant
    
    return resultat

a = calcul_exposant(2, 4)
print ("Le résultat est : ", a)

b = calcul_exposant(12, 2)
print ("Le résultat est : ", b)

