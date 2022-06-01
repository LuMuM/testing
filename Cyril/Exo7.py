# =============================================================================
# Fonction FizzBuzz
# =============================================================================
def fizzbuzz(inf,sup):
    for i in range(inf,sup):
        if i % 3 == 0 and i % 5 == 0:
            print("fizzbuzz", end = " ")
        elif i % 3 == 0:
            print("fizz", end = " ")
        elif i % 5 == 0:
            print("buzz", end = " ")
        else:
            print(i, end = " ")

mini = 1
maxi = 23

fizzbuzz(mini, maxi)
