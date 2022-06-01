#EXO_3 : Écrire une fonction qui retourne tous les multiples 
# d'un nombre compris entre dans un intervalle fourni en entrée.
#Indices : il faudra spécifier un intervalle, recueillir les 
# multiples dans une structure, et tester si un nombre est 
# un multiple d'un autre. Pour cette dernière opération,
# on utilisera le modulo, qui est le reste d'une division euclidienne. 
# Ainsi, 5 % 2 vaut 1, tandis que 4 % 2 vaut 0. Vous l'aurez compris : 
# le modulo d'un nombre par un autre facteur dont il est le multiple vaut 0.


def module():
    var=1
    while var ==1:
        num1 = float(input('choisir un nombre1 (entre -9999~~9999) : '))
        num2 = float(input('choisir un nombre2 (entre -9999~~9999) : '))
        if num1>=-9999 and num1<=9999 and num2>=-9999 and num2<=9999:
            print("si,",num1,"est un multiple de",num2,"?")
            p,q=max(num1,num2),min(num1,num2)
            if q==0:
                print("ne peut pas être calculé")
            else:
                r=p % q
                print ("le module de nombre1: %s et nombre2: %s est %s"%(num1,num2,r))
            if r == 0:
                print("%s est multiple de %s"%(num1,num2))
            else:
                print("donc %s n'est pas multiple de %s"%(num1,num2))
            break
        else:
            print('choisir un nombre (entre -9999~~9999) !!')
    else:
        print('choisir un nombre (entre -9999~~9999) !!')
module()
