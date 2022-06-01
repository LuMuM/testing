def extremites(n):
    i=0
    chaine = ' '
    while i != n:
        i += 1
        chaine += '-'
    return(chaine)

print(extremites(5))



def cotes(l):
    chaine='|'
    for i in list(range(l)):
        chaine +=  ' '
    chaine+='|'
    return(chaine)



def rectangle(l,h):
    print(extremites(l))
    for i in list(range(h)):
        print(cotes(l))
    print(extremites(l))



rectangle(2,2)


