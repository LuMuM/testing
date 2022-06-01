
#creer un carre

def carre(n):
    horiz = "-"
    esp =" "
    print(" ",n*horiz*2," ")
    
    for i in range(n):
        print('|', n*esp*2,'|')
    print(" ",n*horiz*2," ")

carre(int(input("Taille : ")))

#a corriger

"""
str1="abc12"
print('New string variable:',3* str1)
"""