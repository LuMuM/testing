#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 11:56:20 2022

@author: antoinesedoh
"""

def rectangle (largeur,longueur):
    a = " " + longueur * "—" + ""
    print (a)
    for i in range(largeur):
        for j in range(longueur):
            s = "|" + longueur * " "+ "|"   
        #affichage
        print(s)
    print (a)

 
          
rectangle(3,6)
rectangle(4,4)
rectangle(4,6)
rectangle(7,7)
rectangle(17,17)

# x = int (input("Entrez la largeur : "))
# y = int (input("Entrez la longueur : "))
#rectangle(x,y)