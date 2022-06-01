import random


# Fonction pour deviner chiffre aléatoire
def aleat():
    a = random.randint(0, 9)
    b = -1
    while b != a:
        b = int(input("Entrer chiffre entre 0 et 9 : "))
        if a < b:
            print("Le chiffre est inférieur")
        elif b < a:
            print("Le chiffre est supérieur")
    print("Bravo vous avez trouvé le chiffre mystère !")


aleat()