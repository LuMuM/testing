#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:10:35 2022

@author: antoinesedoh
"""


def retrait_voyelle (liste, mot) :
    #la transformer en majuscule
    mot_maj = mot.lower()
    mot_liste = list(mot_maj)
    print (mot_maj)
    print(mot_liste)
    #passer en revue les voyelles,lettre représente "A", puis "E", etc.
    for lettre in mot_liste:        
        if (lettre in liste ):
            mot_liste.remove(lettre)
    return mot_liste


# Liste des voyelles
liste = ['a','e','i','o','u', 'y']
print (liste)
#saisir la chaîne de caractères 
phrase = input("saisir votre mot : ")

#et finalement
resultat = retrait_voyelle(liste, phrase)
mot_post_retrait=''.join([str(i) for i in resultat]) 
print("Le mot après retrait est : " , mot_post_retrait )