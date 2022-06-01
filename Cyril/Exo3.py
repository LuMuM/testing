# =============================================================================
# Fonction qui, pour un nombre spécifié, affiche ses multiples dans un intervalle spécifié
# =============================================================================
def mult (nbr,inf,sup):
    lst = list()
    for i in range(inf,sup+1):
        if nbr % i == 0:
            lst.append(i)
    print(lst)

nombre = 10
inferi = 1
superi  = 6

mult(nombre,inferi,superi)
