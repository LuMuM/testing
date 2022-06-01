#j'ai voulu tester avec une variable avant d'essayer avec un intervalle
def fizzbuzz(x):
    if (x%3 == 0 and x%5 == 0):
        return "fizzbuzz" 
    elif (x%3 == 0):
        return ("fizz" )
    elif (x%5 == 0):
        return ("buzz") 

print(fizzbuzz(15))


#go intervalle
def Fizzbuzz(y):
    for i in y:
        if i % 3 == 0 and i % 5 == 0:
          print("Le nombre",i, "est un Fizzbuzz")
        elif i % 3 == 0:
          print("Le nombre",i, "est un Fizz")
        elif i % 5 == 0:
          print("Le nombre",i, "est un Buzz")
        else:
          print (i)

    

Fizzbuzz(range (1,21))