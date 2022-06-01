
def changer_tab():
    from random import random
    from numpy import diag
    import numpy as np
    from pandas import DataFrame

    liste=[[1,2,3],[4,5,6],[7,8,9]]
    for a in range(len(liste)):
        for b in range(len(liste)):
            if a==b:
                liste[a][b] = "diag"
            if a!=b:
                liste[a][b] = "pas_diag"
    df1=DataFrame(np.array(liste))
    print(df1)

changer_tab()