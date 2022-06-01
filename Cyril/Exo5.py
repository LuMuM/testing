# =============================================================================
# Fonction qui remplace les éléments de la diagonale d'une liste par "diag"
# =============================================================================
def diag (liste):
    for i in range(len(liste)):
        for j in range(len(liste[i])):
            if i == j:
                liste[i][j] = "diag"
    print(liste)

TAB = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
diag(TAB)

# =============================================================================
# Variante avec un dataframe. Décommenter pour essayer
# =============================================================================
  
# from pandas import DataFrame
#
# def diag (df):
#     for i in range(len(df)):
#         for j in range(len(df[i])):
#             if i == j:
#                 df[i][j] = "diag"
#     print(df)
#
# TAB = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# diag(TAB)
