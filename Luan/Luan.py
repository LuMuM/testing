from numpy import sort
from sqlalchemy import true


Luan = list()
print(Luan)

Luan.append(1)
print(Luan)

lmots = ["Jus d'orange", "tomates cerises"]
#Notes Stéphane
lnombre = [17,20,12]
print(lnombre)


lmixte = ["j'ai ", 20, " ans"]
print(lmixte)

lrange =range(20)
for i in lrange:
    print(i)

tey = list(range(20))
print(tey)

toma = list(range(0,51,3))
print(toma)

print(len(toma))

dernier = toma[len(toma) - 1]
print(dernier)
print(toma[-1])

bigbang = ["dino", 3]
orange = ["jus d'orange"]

#Créer une liste composé de listes
megaliste = [bigbang, orange]
print(megaliste)


#megaliste[n. liste][element]
el = megaliste[0][1]

print(el)

#print(globals()), affiche toute les variables 

a = [3,2,1,4,9,3]
print(a)
print(a[-1])

#Trie la liste "a" en ordre croissante
print(sort(a))
print(sort(a)[-1])

#Trie la liste "b" en ordre alphabétique
b = ["cc","bb","c", ""]
print(b)
print(sort(b))

#Supprime l'élément d'index 0
bigbang.pop(0)
print(bigbang)

print(sorted(a, reverse=True))

a.sort(reverse=True)
print(a)

