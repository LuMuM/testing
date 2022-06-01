# Affiche un rectangle dans le terminal
def rectangle(x, y):
    largeur = " "
    space = ""
    for i in range(x):
        largeur += "-"
        space += " "
    print(largeur)
    for j in range(y):
        print("|" + space + "|")
    print(largeur)


rectangle(6, 3)
