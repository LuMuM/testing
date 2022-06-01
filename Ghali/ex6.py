#j'ai du un peu de mal avec celui là, en me documentant je tombe sur des programmes déjà faits mais petite galère à le faire seul

def factorielle(x,y):
    n = x
    for i in range (x,y):
        n = i * (n - 1)
    

def fac(n)

print(factorielle(5,20))


#correction du prof
def factorielle(n):
  # Condition d'arrêt
  if n == 0 or n == 1:
    return 1
  else:
    return n * factorielle(n-1)
  
  
for n in range(0, 11):
  print(factorielle(n), end=" ")