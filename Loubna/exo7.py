intervalle=list(range(1,16))

def multiple(intervalle):

    for i in intervalle:
        if i % 3 == 0 and i % 5 == 0:
             print ("fizzbuzz")
        elif i % 3 == 0 and i % 5 != 0:
            print ("fizz")
        elif i % 5 == 0 and i % 3 != 0:
           print ("buzz")
        else:
            print(i)

multiple(intervalle)s