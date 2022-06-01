#EXO_4 :Jouons à un jeu de plus ou moins. 
# Écrire une procédure dans laquelle un chiffre est sélectionné au hasard entre 0 et 9. 
# L'utilisateur doit deviner lequel c'est. S'il saisit un chiffre supérieur à celui sélectionné, 
# alors la fonction doit afficher que c'est moins, et vice versa. 
# Le jeu s'arrête lorsque l'utilisateur aura deviné le bon chiffre.

def deviner():
    import random
    num = random.randint(0,9)
    i = 1
    while i == 1:
        ans = int(input("Devinez-Vous: "))
        if ans > num:
            print("too big")
        elif ans < num:
            print("too small")
        else:
            print("Deviner Juste!!!")
            break

deviner()      

