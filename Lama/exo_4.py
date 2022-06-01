import random

def jeu():

    nb= -1

    nb_deviné = random.randint(1, 11)

    proche_nb_deviné_plus=nb_deviné+1

    proche_nb_deviné_moins=nb_deviné-1

    print(proche_nb_deviné_plus)

    print(proche_nb_deviné_moins)

 

    while nb != nb_deviné:

        print("choisir un nb entre 1 et 9?")

        nb = int(input())

        if nb==proche_nb_deviné_plus:

            print("Vous etes proche ! et c'est moins")

        if nb==proche_nb_deviné_moins:

            print("Vous etes proche ! et c'est plus") 

        if nb > nb_deviné and nb != proche_nb_deviné_plus and nb!=proche_nb_deviné_moins:

            print("C'est moins.")

        if nb > nb_deviné and nb != proche_nb_deviné_plus and nb!=proche_nb_deviné_moins:

            print("C'est plus.")

        if nb==nb_deviné:

            print("T'as réussi !")

 

jeu()

