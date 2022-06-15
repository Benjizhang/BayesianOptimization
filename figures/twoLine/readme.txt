kind="ei", 
kappa = 2, 
xi=0.5,
kappa_decay=1,
kappa_decay_delay=0
-------------------
def target(x, y):
c = np.exp(-x+y-2)
d = np.exp(x-y-1)
return c+d