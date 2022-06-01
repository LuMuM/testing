#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 11:06:30 2022

@author: antoinesedoh
"""




# Pour les nombres compris dans un intervalle :
# si le nombre est divisible par 3 : on affiche Fizz ;
# si le nombre est divisible par 5 : on affiche Buzz ;
# si le nombre est divisible par 3 et par 5 : on affiche Fizzbuzz ;
# sinon on affiche le nombre.

def fizzbuzz (lower, upper, nombre): 
    for x in range(lower,upper+1, 1):
        if (nombre % 3 == 0 and nombre % 5 == 0):
            print("Fizzbuzz")
            break
        elif(nombre % 3 == 0): 
            print ("Fizz")
            break
        elif(nombre % 5 == 0):
            print("Buzz")
            break
        else:
            print("nombre")
            break
min = 1
max = 20


#Pour tester la fonction
fizzbuzz(min,max, 1)
fizzbuzz(min,max, 2)
fizzbuzz(min,max, 3)
fizzbuzz(min,max, 4)
fizzbuzz(min,max, 5)
fizzbuzz(min,max, 6)
fizzbuzz(min,max, 7)
fizzbuzz(min,max, 8)
fizzbuzz(min,max, 9)
fizzbuzz(min,max, 10)
fizzbuzz(min,max, 11)
fizzbuzz(min,max, 12)
fizzbuzz(min,max, 13)
fizzbuzz(min,max, 14)
fizzbuzz(min,max, 15)
fizzbuzz(min,max, 16)
fizzbuzz(min,max, 17)
fizzbuzz(min,max, 18)
fizzbuzz(min,max, 19)
fizzbuzz(min,max, 20)