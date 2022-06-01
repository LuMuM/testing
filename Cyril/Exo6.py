# =============================================================================
# Fonction factorielle
# =============================================================================

def factorielle(nbr):
    if nbr == 1 or nbr == 0:
        return 1
    else:
        return nbr * factorielle(nbr-1)

n = 7
print(factorielle(n))

for n in range(0,11):
    print(factorielle(n),end = " ")
