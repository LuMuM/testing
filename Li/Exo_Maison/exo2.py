#EXO_2 : Écrire une fonction qui, à chaque chaîne de caractères, 
# retirera les lettres dans une liste fournie en entrée (les voyelles, par ex.).
#Indice : il nous faudra stocker toutes les lettres d'une part, et toutes lettres que 
# l'on souhaitera retenir d'autre part, puis retourner la liste
# des lettres que l'on retiendra.
#Exemple : "voyelles" doit devenir "vlls" si l'on décide d'en retirer les voyelles, 
# avec y inclus.


def supr_voye():
        from curses.ascii import isalpha
        import re
        word = input("taper un mot ici : ")
        if word.isalpha() == True :
                wordlower = word.lower()
                print('Convertir toutes les lettres en minuscules : ',wordlower)
                newword = re.sub(r'a|e|i|o|u|y','',wordlower)
                print('supprimer toutes les voyelles : ',newword)
        else:
                print("Les caractères autres que les lettres ne sont pas autorisés")

supr_voye()