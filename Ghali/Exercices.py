## Exercice 0
def Puissance (x, y):
    print(x**y)

#(test)
Puissance(5,2) #(test)

## Exercice 1
def Boucle (mot):
    while (not mot =="stop"):
        mot = input("Oh non, voici chipeur, arrête le en disant stop ")
print("HELLO FOLKS, IT S ME, DOOOOORA ")
Boucle("La drogue c'est mal")

## Exercice 2

def lettre ():
    mota = input ("Choisissez un mot")
    lettreb = input ("Retirez une lettre")
    listeee = []
    for i in range (0, len(mota)):
        listeee.append(mota[i])
        if (mota[i] == lettreb) :
            listeee[i] = []
            print (listeee)



lettre()





## Exercice 3
def Multiples():
    start = input ("début de l'intervalle")
    stop = input ("fin de l'intervalle")
    step = input ("Multiple de ?")
    print("Les multiples de ", step ," sur l'intervalle [",start,"," ,stop,"] sont :")    
    stop +=  1
    for i in range (start,stop,step):
        print (i)

#test
Multiples()

## Exercice 4
import random
from tkinter import Y
def Akinator():
    x = random.randint(0, 9)
    print(x)
    y = input("Choisissez une valeur y ")
    while (not x == y):
        if (x <= y):
            y = input("Trop haut pelo")
        elif (x >= y):
            y = input("Trop bas pelo")
    
    if (x == y):
        print("bien joué champion") 


Akinator()
