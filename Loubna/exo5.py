tableau = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

def passage(tableau):
    for i in range(len(tableau)):
        for j in range(len(tableau)):
            if i == j: 
                tableau[i][j] = "diag"
    return tableau
    


print("tableai initial: \n",tableau)
print("resultat:\n" ,passage(tableau))

#---------------------------------------------------------------

tableau = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

def passage(tableau):
    for i in range(len(tableau)):
        for j in range(len(tableau)):
            if i != j: 
                tableau[i][j] = "non diag"
    return tableau
    


print("tableai initial: \n",tableau)
print("resultat:\n" ,passage(tableau))