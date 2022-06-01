#----- exercice 5

# Creons la fonction

def remplace(tableau):
    for i in range(len(tableau)):
        for j in range(len(tableau[i])):
            if (i == j):
                tableau[i][j] = "diag"
    #affichage1
    print()
    print("Transormation")
    print(tableau)


#creons un tableau

print("Entrez le éléments du tableau")
tabl = [[input("a1,1 : "),input("a1,2 : "),input("a1,3 : ")],[input("a2,1 : "),input("a2,2 : "),input("a2,3 : ")],[input("a3,1 : "),input("a3,2 : "),input("a3,3 : ")]]
print(tabl)


remplace(tabl)

# Partie 2 - pas diag
def remplace_bis(tableau):
    for i in range(len(tableau)):
        for j in range(len(tableau[i])):
            if (i == j):
                tableau[i][j] = "diag"
            else:
                tableau[i][j] = "pas diag"
    #affichage1
    print()
    print("Transormation")
    print(tableau)


remplace_bis(tabl)