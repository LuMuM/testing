import random

# -----------{ Exercice 0 }----------- 
print("Exercice 0")

def expo (a,b):         #Création procédure: a exposant b
    return a**b

b=1 #Changer b pour choisir l'exposant

for a in range (1,11):
    print(expo(a,b))



# -----------{ Exercice 1 }----------- 
print("Exercice 1")

x=0
while x==0:
    y = input()
    if y== "stop":
        x=1                 #x n'est plus égal à 0. La boucle s'arrete.



# -----------{ Exercice 3 }-----------
print("Exercice 3")

def mult(a,b,c):                # Les multiple de c dans (a,b)
    Liste = []
    x=list(range(a,b))
    for i in x:
        if i % c == 0:
         Liste.append(i)
    return Liste

print(mult(int(input()),int(input()), int(input())))



# -----------{ Exercice 4 }-----------
print("Exercice 4")

def jeu():
    bon=random.randint(0,9)
    win=False                                   #Tant que Win n'est pas True, le jeu continue
    while win==False:
        dem = int(input())
        if dem>bon:
            print ("Le chiffre est plus petit")
        else:
            if dem<bon:
                print ("Le chiffre est plus grand")   
            else:
                print("Win")
                win = True                     #Win devient True uniquement sous la condition dem == bon
jeu()

# -----------{ Exercice 5 }-----------

print("Exercie 5")

Tab=[[1,2,3],[1,2,3],[1,2,3]] 

def Diag ():
    x=0
    while x<len(Tab):                   # x augment jusqu'à avoir fait tout le tableau
        Tab[x][x]="diag"                
        x+=1 
    print(Tab)

Diag()


print("Exercie 5bis")
def PasDiag (Tab):
    for i in range (len(Tab)):
        for j in range (len(Tab)):
           if (i!=j):
                Tab[i][j]="Pas Diag"
    print(Tab)

print(PasDiag(Tab))
