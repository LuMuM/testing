# =============================================================================
# Fonction de mise à la puissance d'un nombre entre 1 et 10
# =============================================================================

def power(elem, exp):
    if elem < 1 or elem > 10:
        print("Erreur, entrer un nombre compris entre 1 et 10")
    else:
        power = elem**exp
        return power

power(10,3)
