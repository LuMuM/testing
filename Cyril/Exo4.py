# =============================================================================
# Fonction : jeu du plus-ou-moins
# =============================================================================
import random as rd

def plus_ou_moins ():
    x = str(rd.randint(0, 9))
    c = 0 # Compteur de coups
    trial = input("Entrez une valeur entre 0 et 9 : ")
    while trial != x:
        c += 1
        if trial > x:
            trial = input("Moins")
        elif trial < x:
            trial = input("Plus")
    else:
        print("Gagné en " + str(c) + " coups ! La valeur était " + x)
    
plus_ou_moins()