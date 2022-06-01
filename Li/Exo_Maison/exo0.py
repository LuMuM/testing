#EXO_0 : Écrire une fonction dans un module Python qui, 
# à tout chiffre/nombre compris entre 1 et 10, 
# lui appliquera un exposant que l'on spécifiera. 
# Par exemple, si l'on souhaite 10 exposant 3, 
# nous obtenons 1000.

exposant = float(input("le exposant :"))
a = float(input("choisi une nombre entre 1 et 10 : "))
if a >= 1 and a <= 10 :
    print (a**exposant)
 
    