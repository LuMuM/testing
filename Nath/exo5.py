
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

#On remplace les éléments de la diagonale par "pas_diag"
def remplacer_pas_diag(tableau):
    for ligne in range(0,len(tableau)):
        for colonne in range(0,len(tableau)):
            if ligne!=colonne:
                tableau[ligne][colonne]="pas_diag"
            
    return(tableau)
print(remplacer_pas_diag(tableau))



