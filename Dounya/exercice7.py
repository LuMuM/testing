# Fonction qui prend un intervalle en entrée et qui résout le problème Fizzbuzz
def fizzbuzz(min, max):
    for i in range(min, max+1):
        if i % 3 == 0 and i % 5 == 0:
            print("Fizzbuzz")
        elif i % 3 == 0:
            print("Fizz")
        elif i % 5 == 0:
            print("Buzz")
        else:
             print(i)


fizzbuzz(1, 20)
