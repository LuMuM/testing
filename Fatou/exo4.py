# -*- coding: utf-8 -*-
"""
Created on Sat May 22 00:04:12 2022

@author: HP
"""

import random


def random_():
    
  a = -1
  b = random.randint(0, 9)
  # On pourrait aussi faire un while True et faire un break plus loin, comme ci-dessus.
  while a!= b:
      a = int(input("Saisir un chiffre entre 0 et 9:"))
    if a > b:
      print("Le chiffre est plus petit , inferieur")
    elif a < b:
      print("Le chiffre est plus grand , superieur")
  print("Bravo vous avez trouvé le bon chiffre !")
  
random_()