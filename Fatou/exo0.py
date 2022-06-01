# -*- coding: utf-8 -*-
"""
Created on Sat May 21 23:56:37 2022

@author: HP
"""

 
def exposant(nbre,exposant) :
   
  
    print ("Nombre :", nbre)
    print ("Exposant :", exposant)

    # On effectue le calcul 
    #for i in range(0,exposant+1):
    resultat = nbre ** exposant

    return resultat

X = exposant(4,8)
print ("Le résultat est : ", X)

