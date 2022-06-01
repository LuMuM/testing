# -*- coding: utf-8 -*-
"""
Created on Mon May 23 10:31:50 2022

@author: HP
"""

def diagonale(tableau):

    
    for i in range(len(tableau)): #parcours les i valeurs du tableau
        for j in  range(len(tableau)) :# parcours les j valeurs du tableau
            if i == j: #verifie a chaque fois si i=j
                tableau[i][j]= "diag"  
    return tableau    
        
tableau= [ [1, 2, 3], [4, 5, 6], [7, 8, 9]]     

print(diagonale(tableau))
