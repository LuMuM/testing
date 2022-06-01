intervalle= list(range(1,21)) #liste des nombre de 1 à 20
print(intervalle) 
def fizzbuzz (intervalle): #fonction prend l'intervalle dans le parametre pour le traiter selon les regles de jeu
     for i in intervalle:  #pour chaque element dans la liste intervalle
        if i%3==0 and i%5!=0:  #si le nombre est divisible par 3 mais pas deivisible par 5 alors on affiche Fizz
            print ("Fizz")
        elif i%5==0 and i%3!=0:  #si le nombre est divisible par 5 mais pas deivisible par 3 alors on affiche Buzz
            print ("Buzz")
        elif i%5==0 and i%3==0 :   #si le nombre est divisible par 5 et deivisible par 3 alors on affiche Fizzbuzz
            print ("Fizzbuzz")
        else:
            print (i)  #sinon on affiche le nombre 
    
print (fizzbuzz(intervalle))
       
        
        
        
            