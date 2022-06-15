kind="ei", 
kappa = 2, 
xi=0.5,
kappa_decay=1,
kappa_decay_delay=0
————————————————
def target(x, y):
b = np.exp(-x+y)
return b