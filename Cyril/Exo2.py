# =============================================================================
# Fonction de suppression de lettres dans une chaîne de caractère
# =============================================================================
def rmv_letter (chaine, liste):
    new_str = ''
    for i in range(0,len(chaine)+1):
        x = 0
        for l in liste:
            if chaine[i:i+1] != l:
                x += 1
            if x == len(liste):
                new_str = new_str + chaine[i:i+1]
                
    return new_str


word = 'papyto'
lst = ["a","e","i","o","u","y"]

rmv_letter(word, lst)
