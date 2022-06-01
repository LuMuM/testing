# Les exercices maison

Attention aux terminologies : une fonction et une procédure se ressemblent beaucoup, mais divergent sur un point ;-)

## Exercice 0

Écrire une fonction dans un module Python qui, à tout chiffre/nombre compris entre 1 et 10, lui appliquera un exposant que l'on spécifiera. Par exemple, si l'on souhaite 10 exposant 3, nous obtenons 1000.

## Exercice 1

Écrire une procédure dont l'exécution ne cessera que lorsque l'utilisateur entrera la chaîne de caractères (string) suivante : `"stop"` .

Astuce : la fonction à utiliser pour saisir une donnée est `input()` .


## Exercice 2

Écrire une fonction qui, à chaque chaîne de caractères, retirera les lettres dans une liste fournie en entrée (les voyelles, par ex.).

Indice : il nous faudra stocker toutes les lettres d'une part, et toutes lettres que l'on souhaitera retenir d'autre part, puis retourner la liste des lettres que l'on retiendra.

Exemple : "voyelles" doit devenir "vlls" si l'on décide d'en retirer les voyelles, avec y inclus.

## Exercice 3

Écrire une fonction qui retourne tous les multiples d'un nombre compris entre dans un intervalle fourni en entrée.

Indices : il faudra spécifier un intervalle, recueillir les multiples dans une structure, et tester si un nombre est un multiple d'un autre. Pour cette dernière opération, on utilisera le modulo, qui est le reste d'une division euclidienne. Ainsi, `5 % 2` vaut `1`, tandis que `4 % 2` vaut `0`. Vous l'aurez compris : le modulo d'un nombre par un autre facteur dont il est le multiple vaut 0.

## Exercice 4

Jouons à un jeu de plus ou moins. Écrire une procédure dans laquelle un chiffre est sélectionné au hasard entre 0 et 9. L'utilisateur doit deviner lequel c'est. S'il saisit un chiffre supérieur à celui sélectionné, alors la fonction doit afficher que c'est moins, et vice versa. Le jeu s'arrête lorsque l'utilisateur aura deviné le bon chiffre.

Astuce : pour sélectionner un chiffre entre 0 et 9, on devra importer un package et utiliser une fonction de ce dernier, tel que ci-dessous :


```python
import random


random.randint(0, 9) # Ceci doit faire partie du corps de la procédure à développer.
```

```
## 6
```

## Exercice 5

Créer une fonction qui remplace tous les termes dans la diagonale d'un tableau carré par la chaîne `"diag"`. Voici un exemple de tableau et la sortie attendue pour celui-ci :

*Entrée :*

|     |     |     |
|-----|-----|-----|
| 1   | 2   | 3   |
| 4   | 5   | 6   |
| 7   | 8   | 9   |

*Sortie :*

|      |      |      |
|------|------|------|
| diag | 2    | 3    |
| 4    | diag | 6    |
| 7    | 8    | diag |

Sortie version Python :

``` python
[["diag", 2, 3], [4, "diag", 6], [7, 8, "diag"]]
```

Astuces :

-   en Python, un tableau est une liste qui contient des listes, ou une liste de listes. On peut y mettre tous les éléments que l'on souhaite, et même les panacher, et on peut avoir des lignes avec un nombre d'éléments variable . Nous nous en tiendrons à un tableau carré de 9 éléments que l'on remplira d'éléments quelconques.

-   soit L une liste, la longueur d'une liste s'obtient ainsi : `len(L)`

## Exercice 6

La fonction factorielle est certainement la [fonction récursive](https://www.jesuisundev.com/comprendre-la-recursivite-en-7-min/) la plus simple à comprendre. Une fonction récursive signifie que l'on peut appeler la fonction que l'on définit dans le corps de son expression, en décomposant la donnée à traiter en blocs plus petits. Concrètement, si l'on devait définir la factorielle récursivement, on ferait ainsi :

``` python
factorielle(n) = n * factorielle(n-1)
```

On demande ici de coder la fonction factorielle.

Indice : il faudra forcément une condition d'arrêt.

## Exercice 7

Voici un problème populaire : le fizzbuzz. Je vous propose d'en faire une version basique (il y a des concours d'implémentations sur internet !). Les données du problème sont les suivantes :

Pour les nombres compris dans un intervalle :

-   si le nombre est divisible par 3 : on affiche `Fizz` ;

-   si le nombre est divisible par 5 : on affiche `Buzz` ;

-   si le nombre est divisible par 3 et par 5 : on affiche `Fizzbuzz` ;

-   sinon on affiche le nombre.

Codez une procédure qui prend un intervalle en entrée (mettons de 1 à 20, intervalle fermé) et qui résout ce problème.
