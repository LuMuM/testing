#Exo 0
def Fonction1(a,b) :
    if   1 <= a <= 10 :
        x= a**b
        print(x)

Fonction1(10,3)

#Exo 1
x=1
print("ecrire 'stop' pour cesser l'execution")
while x != "stop":
    x=input()

#Exo 2
#Exo 3

#Exo 4
import random

print("choisir un chiffre entre 0 et 9")
#def Plus_Moins(x):
#    choix = random.randint(0, 9)
#    while x != choix:
#       x=input()

#print(Plus_Moins(x))

#Exo 5
tableau=[[1,2,3],[4,5,6],[7,8,9]]
print(tableau)

#On remplace les éléments de la diagonale par "diag"
def remplacer_diag(tableau):
    for ligne in range(0,len(tableau)):
        for colonne in range(0,len(tableau)):
            if ligne==colonne:
                tableau[ligne][colonne]="diag"
            
    return(tableau)
print(remplacer_diag(tableau))

#Exo 6
#Utilisation de fonction récursive (appeler la fonction dans la fonction)
def Factorielle(n):
    if n==0:
        return 1
    else:
        return n*Factorielle(n-1)

nombre= int(input("Entrez un entier"))
print(Factorielle(nombre))

#Exo 7
#Modulo = pour donner le diviseur d'un dhiffre donné
def Fizz_Buzz(y):
    for i in y:
        if i % 3==0 and 1 %5==0:
            print("Fizz_Buzz")
        elif i % 3==0:
            print("Fizz")
        elif i % 5==0:
            print("Buzz")
        else:
            print(i)
        
print(Fizz_Buzz(range(1,21)))
