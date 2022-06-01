# Fonction retire les lettres du mot
def retire_voyelle(mot, lettres):
    res = ""
    for x in mot:
        if x not in lettres:
            res += x
    print(res)


voyelles = ["a", "e", "i", "o", "u", "y"]
retire_voyelle("bonjour", voyelles)
