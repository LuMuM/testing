def carré(x):
    print ("--"*x)
    for i in range (x) :

        print ("|"," "*x,"|")
    print ("--"*x)

carré(5)

def rectangle(longueur, largeur):
    print ("","___"*(largeur-1))
    for i in range (longueur) :
        if i == longueur - 1 :
            print ("|","__"*largeur,"|")
        
        elif (not i ==  longueur - 1) :
            print ("|","  "*largeur,"|")   

rectangle(0,6)

