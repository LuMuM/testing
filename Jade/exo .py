
#exo5

tableau=[[1,2,3],[4,5,6],[7,8,9]]
diag = 'diag'


def diagonale(tableau, diag):
		
	for i in range(0,len(tableau)):
		for j in range(0,len(tableau)):
		    if i==j:
		        tableau[i][j] = diag
		    else:
		        tableau[i][j] = 'pas diag'
	return tableau
		
print(diagonale(tableau,diag))