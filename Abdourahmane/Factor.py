#exercice 7

def factorielle(n):

    fact = n
    while not(n==1):
        fact = fact * (n - 1)
        n = n - 1
    print(fact)

factorielle(int(input("Entier : ")))


# il fallait utiliser la récursivité