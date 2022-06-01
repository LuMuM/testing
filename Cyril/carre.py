def square(hauteur, largeur):
    for i in range(hauteur):
        if i == 0 or i == hauteur-1:
            for j in range(largeur):
                if j == largeur-1:
                    print(" -")
                else:
                    print(" -", end = "")
        else:
            for j in range(largeur):
                if j == 0:
                    print("| ", end = "")
                elif j == largeur-1:
                    print("  |")
                else:
                    print("  ", end = "")

square(5,5)