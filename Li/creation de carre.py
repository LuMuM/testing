from ctypes.wintypes import HHOOK


def carre(n):
    larger="-"
    space=" "
    n = int(input("choisi le dimsion: "))
    print("|",n*larger,"|")
    for i in range(n):
        print("|",n*space,"|")
    print("|",n*larger,"|")
carre(2)






