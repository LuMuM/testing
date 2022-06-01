# Fonction qui retourne tous les multiples d'un nombre compris dans un intervalle fourni en entrée
def multiples(nbr, min, max):
    l = []
    for i in range(min, max+1):
        if i % nbr == 0:
            l.append(i)
    print(l)


multiples(2, 1, 30)
