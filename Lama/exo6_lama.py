print("entrez le nombre que vous voulez faire factorielle:") #demander à l'utilisateur de choisir le nombre à faire le factorielle
n=int(input()) #le stocker
def factorielle(n):
    if n==0:
        return 1   #si n est 0 alors le factorielle est 1 et on s'arrete et on rentre pas dans else
    else:
        return n* factorielle(n-1) #le factorielle boucle jusqu'à n=0 elle rentre dans le condition precedent sans boucler donc on obtient le resultat en bouclant mais dés qu'on arrive à 0 on s'arrete
print(factorielle(n))

