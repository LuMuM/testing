intervalle=list(range(1,11))

def multiple(intervalle, nb):
    multiples = []
   
    for i in intervalle:
        if i % nb == 0:
            multiples.append(i)#ajouter le nombre au tableau 
    return multiples

#print("l'intervalle est: ",intervalle)
#print("on va sortir les chiffres multiples par 2 :")

print("Choisir un nombre?")
nb = int(input())
print("la sortie est :", multiple(intervalle,nb))