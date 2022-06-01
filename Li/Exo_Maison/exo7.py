# si le nombre est divisible par 3 : on affiche Fizz ;
# si le nombre est divisible par 5 : on affiche Buzz ;
# si le nombre est divisible par 3 et par 5 : on affiche Fizzbuzz ;
# sinon on affiche le nombre.

def fizzbuzz(a):
    for num1 in a:
        print(num1)
        r = num1 % 3
        t = num1 % 5
        # print ("%s / 3 = %s"%(num1,num1/3))
        # print ("%s / 5 = %s"%(num1,num1/5))
        if r == 0 and t!= 0:
            print("Fizz")
        if t == 0 and r!=0: 
            print("Buzz")
        if t == 0 and r == 0:
            print("FizzBuzz")    
print(fizzbuzz(range(0,21)))   

            