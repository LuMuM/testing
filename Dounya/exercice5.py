import numpy as np


# Fonction qui remplace tous les termes dans la diagonale d'un tableau carré par la chaîne "diag"
def remplace_diag(tableau):
    for liste in tableau:
        for val in liste:
            indiceLigne = tableau.index(liste)
            indiceCol = liste.index(val)
            if indiceLigne == indiceCol:
                tableau[indiceLigne][indiceCol] = "diag"
    return np.array(tableau)

# Fonction qui remplace tous les termes pas dans la diagonale d'un tableau carré par la chaîne "pas diag"
def remplace_pasdiag(tableau):
    for i in range(len(tableau)):
        for j in range(len(tableau)):
            if i != j:
                tableau[i][j] = "pas diag"
    return np.array(tableau)


tab = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(remplace_diag(tab))
tab = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(remplace_pasdiag(tab))

