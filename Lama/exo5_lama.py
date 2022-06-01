tableau=[[1,2,3],[4,5,6],[7,8,9]]
print(tableau)

def passage(tableau): 
    for i in range(len(tableau)):  #passer par le length de tableau c'est à dire i =lignes
        for j in range(len(tableau)):#puis par colonne de i c'est à dire j
            if i==j:
                tableau[i][j]= "diag"  #remplacer la case de diagonale c'est à dire quand i==j par le mot "diag"
    return tableau

print(passage(tableau))


#----------------l'exo inverse----------------------------------#

tableau=[[1,2,3],[4,5,6],[7,8,9]]
print(tableau)

def passage(tableau): 
    for i in range(len(tableau)):  #passer par le length de tableau c'est à dire i =lignes
        for j in range(len(tableau)):#puis par colonne de i c'est à dire j
            if i!=j:
                tableau[i][j]= " not diag"  #remplacer la case de diagonale c'est à dire quand i not egale à j par le mot "not diag"
    return tableau

print(passage(tableau))


