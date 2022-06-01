# -*- coding: utf-8 -*-
"""
Created on Sun May 22 00:00:29 2022

@author: HP
"""

def retire_voyelles (mots, lettres):
    
    res = ""
    for i in mots:
        if i not in lettres:
            res =res + i
    print(res)


voyelles = ["a", "e", "i", "o", "u", "y"]
retire_voyelles ("Salut je m'appelle Fatou", voyelles) 