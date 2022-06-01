def factorielle(n):
    if n == 0:
        return 1 #si n=0 alors on arrete la boucle 
    else:
        return n  * factorielle(n-1)#si n n'est pas 0 alors on rappelle la fonction jusque à que n =0


print("entrez un nombre?")
n=int(input())
print(factorielle(n))

