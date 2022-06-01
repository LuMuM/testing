# Exercice 0

#Écrire une fonction dans un module Python qui, à tout chiffre/nombre compris entre 1 et 10, lui appliquera un exposant que l'on spécifiera. Par exemple, si l'on souhaite 10 exposant 3, nous obtenons 1000.#

def calcul_exposant(nombre, exposant) :
    print ("****Afficher les resultats ****")
    print ("Nombre :", nombre) #Afficher le nombre
    print ("Exposant :", exposant) #Afficher l'exposant

    # affichage du resultat 
    resultat = nombre ** exposant

    return resultat


#Calcul
x = calcul_exposant(3, 9)
print ("Le résultat est : ", x)

y = calcul_exposant(36, 6)
print ("Le résultat est : ", y)

z = calcul_exposant(25,5)
print ("Le résultat est : ", z)

