# -*- coding: utf-8 -*-
"""
Created on Mon May 23 00:05:31 2022

@author: HP
"""

def multiple(intervalle, facteur):
  multiples = []
  for i in intervalle:
    if i % facteur == 0:
      multiples.append(i)
  return multiples


print(multiple(range(2, 31), 2))