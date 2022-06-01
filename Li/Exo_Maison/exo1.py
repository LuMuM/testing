#EXO_1 :Écrire une procédure dont l'exécution ne cessera que 
# lorsque l'utilisateur entrera la chaîne de caractères (string) 
# suivante : "stop" . Astuce : la fonction à utiliser 
# pour saisir une donnée est input() .

var=1
while var ==1:
    user = input("taper un mot ici : ")
    if user =='stop':
        print('exit')
        exit()
        break
    else:
        print('rattaper un mot!!')
else:
    print('rattaper un mot!!')




