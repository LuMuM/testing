from enum import Enum
class direction(Enum):
    HORIZ = "horizontal",
    VERT = "vertical"


def carre (dim) : 
    print(' ' + '__'* dim +'  ' )
    for j in range(dim):
        chain = "|"
        for i in range(dim) : 
            chain += '  '
            if i == dim - 1 :
                chain += '|'
        print(chain)
        chain = ''
    print(' ' + '--'* dim +'  ' )
    
def rectangle (long, direction) : 
    
    
    if direction == 'horizontale' : 
        vert = 1
        horiz = 2
    else :
        vert = 2
        horiz = 1

    print(' ' + '__'* dim*horiz +'  ' )
    for j in range(dim*vert):
        chain = "|"
        for i in range(dim*horiz) : 
            chain += '  '
            if i == (dim*horiz) -1 :
                chain += '|'
        print(chain)
        chain = ''
    print(' ' + '--'* dim*horiz +'  ' ) 

a = int(input())
carre(a)
rectangle(a,'horizontale')


    
