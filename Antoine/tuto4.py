#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 11:43:28 2022

@author: antoinesedoh
"""
def CarteEtu (nom, prenom, numetu) :
    print ("")
    print ("--- Carte étudiante ---")
    print ("Nom :", nom)
    print ("Prénom :", prenom)
    print ("Numéro étudiant :", numetu)
    
def moyenne (note1, note2, note3, note4, note5, nom) :
    print ("")
    print ("--- Caalcul de la Moyenne ---")
    print ("La première note est de : ", note1)
    print ("La deuxième note est de : ", note2)
    print ("La troisième note est de : ", note3)
    print ("La quatrième note est de : ", note4)
    print ("La cinquième note est de : ", note5)
    somme = note1 + note2 + note3 + note4 + note5 
    moyenne = somme / 5
    print ("La moyenne de : ",nom, "est de : ",moyenne)


p = "Antoine"
n = "SEDOH"
numero_etudiant= 345089

CarteEtu(n, p, numero_etudiant)
CarteEtu("James", "Wilson",  145687)

moyenne(12, 18, 15, 16, 17, "James")
#print ("La moyenne de ", note1, note2, note3, note4, note5, "est", moyenne )