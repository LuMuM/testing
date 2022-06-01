#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 13:59:32 2022

@author: antoinesedoh
"""

# Écrire une procédure dont l'exécution ne cessera que lorsque l'utilisateur entrera la 
# chaîne de caractères (string) suivante : "stop" .
def saisie() :
    liste = []
    #saisie
    saisie = str(input("Saisie : "))
    while saisie != "stop" :
        liste.append (saisie)
        # On redemande la saisie        
        saisie = str(input("Saisie : ")) 
    print("Fin")

    print ("Les éléments rentrés sont : ")
    print (liste)

saisie()
