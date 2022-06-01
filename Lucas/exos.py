import random
from pandas import DataFrame as df
import pandas as pd
import numpy as np
#Exo 0

def puissance (nombre, exposant):
    if nombre >= 1 and nombre <= 10 :
        return (nombre**exposant)
    else : 
        print('le premier nombre doit etre compris entre 1 et 10')

print(puissance(int(input()), int(input())))
    
#Exo 1

def procedure() : 
    i = False

    while not(i) :
        a = input()
        if a == 'stop' :
            i = True

procedure()

#Exo 2

def enlever(liste, mot) :  
    for a in mot : 
        for j in liste :
            if a == j :
                print(a, 'est dans la liste : on le supprime')
                mot = mot.replace(a,'') 
    print(mot)           
    
        
   


voyelle = ['a', 'e', 'i', 'o', 'u', 'y']
enlever(voyelle, input())

#Exo 3

def multiple(nombre, min, max) :
    a = list()
    i = min
    while i <= max :
        if i != 0 :
            if i%nombre == 0 :
                a.append(i)

        i += 1

    print(a)

multiple(int(input()),int(input()), int(input()) )

#Exo 4


def guess() :
    sol = random.randint(0,9)
    trouve = False
    while not(trouve) : 
        essai = int(input())
        if essai == sol :
            print('trouve !')
            trouve = True
        else :
            if essai < sol :
                print('plus grand !')
            else :
                print('plus petit')

guess()




#Exo bonus 

def split(chaine, separateur) :
    liste = list()
    mot = str()
    j = 0
    while j < len(chaine) :
        if chaine[j] != separateur :
            mot += chaine[j]
        if chaine[j] == separateur or j == len(chaine) - 1:
            liste.append(mot)
            mot = ''
        j += 1
    return liste

print(split(input(),','))


#Exo 5

#On definit la fonction 
#qui s'applique sur une superliste

def remplace(tab, mot1, mot2):
    #On recupere les dimensions du tableau  
    dim = len(tab)

    #1ere boucle : toutes les valeurs qui ne sont 
    #pas dans la diagonales sont remplacées

    for i in range(dim):
        for j in range(dim):
            if i != j :
                tab[i][j] = 'pasdg'

    #2eme boucle : toutes les valeurs de la diagonales 
    #Sont remplaces par mot1
    #et toutes les valeurs de la contre diagonale
    #sont remplacés par le mot2

    i = 0            
    while i < dim:
        if i%2 == 0 :
            tab[i][i] = mot1
            tab[dim - 1- i][i] = mot1
        else :
            tab[i][i] = mot2
            tab[dim - 1- i][i] = mot2
        i += 1


# En entree on recupere la dimension de la matrice carré
print('quelle dimension pour le tableau ? : ')
a = int(input())

#On remplit le tableau sous forme de liste
tab = list()
for i in range(0,a) :
    tab.append([]) 
    for j in range(0,a) :
        tab[i].append(random.randint(0,10))

#On appelle la fonction
remplace(tab, 'dino', 'trip')

#On transforme la superliste en tableau pour l'affichage
df1 = df(np.array(tab))
print(df1)


#Exo 6

def factorielle(nombre):
    if nombre == 0 : 
        return 1
    else :
        return nombre * factorielle(nombre-1)

print(factorielle(int(input())))