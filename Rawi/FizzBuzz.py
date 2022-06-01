def FizzBuzz (a):
    for x in range (1,a):
        if x%3==0 and x%5==0:
         print ("FizzBuzz")
        elif x%3 == 0 :
            print ("Fizz")
        elif x%5 == 0:
          print ("Buzz")
        else:
           print(x)

FizzBuzz (21) #Pour faire de 1 à 20
