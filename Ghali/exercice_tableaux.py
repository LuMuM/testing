#remplacer la diagonale par "diag"


Tableau = [[1,1,1],[2,2,2],[3,3,3]]
print (Tableau)

def diag(x):
    for i in range(len(x)):
    #boucle for sur les lignes du tableau
        for j in range (len(x)):
        #boucle for sur les colones du tableau
            if (i == j):
            #si i est différent de j, alors:
                x[i][j] = "diag"
    return(x)
        

print (diag(Tableau))


#-------------------#
#remplacer tout sauf la diagonale par "pas diag"
def not_diag(x):
    for i in range(len(x)):
    #boucle for sur les lignes du tableau
        for j in range (len(x)):
        #boucle for sur les colones du tableau
            if (i != j):
            #si i est différent de j, alors:
                x[i][j] = "pas diag"
    return(x)

        
print (not_diag(Tableau))