
#--------------------------------------------
size = int(input('Choisir la taille du rectangle?  '))
char  = str(input('Choisir le character pour dessiner?  '))

colonne=1
while colonne <= size:
    if colonne==1 or colonne == size:#pour dessiner la premiere et derniere colonne 
        print(char, char, char, char,char)
        colonne = colonne + 1
    else:
        print(char," ", " "," ", char)#pour dessiner le corps
        colonne = colonne + 1