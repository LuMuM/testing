# =============================================================================
# Fonction qui s'arrête lorsqu'on rentre "stop" dans la console
# =============================================================================
def infinite_input():
    inp = ''
    while (inp != 'stop'):
        print('Programm running...')
        inp = input('stop pour arrêter l''exécution : ')
    else:
        print('Fin du programme')

infinite_input()