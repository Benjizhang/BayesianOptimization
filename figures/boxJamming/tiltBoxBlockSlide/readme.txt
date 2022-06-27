kind="ei", 
kappa = 2, 
xi=0.5,
kappa_decay=1,
kappa_decay_delay=0
——————————————————————
def target(x, y):
    a = np.exp(x+y-7)
    b = np.exp(-x-y+5)
    c = np.exp(-x+y-2)
    d = np.exp(x-y-1)
    return 4-(a+b+c+d)