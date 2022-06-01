
def factorielle():
    a = int(input('choisi un numero ici:'))
    num = 1
    if a < 0:
        print("Les nombres négatifs n'ont pas de factorielle")
    elif a == 0:
        print("La factorielle de 0 est 1")
    else:
        for i in range(1,a + 1):
            num *= i
    print(num)

factorielle()