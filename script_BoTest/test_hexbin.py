import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

def target(x, y):
    a = np.exp(-( (x - 2)**2/0.7 + (y - 4)**2/1.2) + (x - 2)*(y - 4)/1.6 )
    b = np.exp(-( (x - 4)**2/3 + (y - 2)**2/2.) )
    c = np.exp(-( (x - 4)**2/0.5 + (y - 4)**2/0.5) + (x - 4)*(y - 4)/0.5 )
    d = np.sin(3.1415 * x)
    e = np.exp(-( (x - 5.5)**2/0.5 + (y - 5.5)**2/.5) )
    # return 2*a + b - c + 0.17 * d + 2*e
    return -x ** 2 - (y - 4) ** 2 + 4



n = 1e5
# x = y = np.linspace(0, 6, 300)
x = y = np.linspace(0, 6, 100)
X, Y = np.meshgrid(x, y)
x = X.ravel()
y = Y.ravel()
print(x)
print(np.vstack([x, y]))
# X = np.vstack([x, y]).T[:, [1, 0]]
X = np.vstack([x, y])
z = target(x, y)



print(max(z)) # 4 for f = lambda x,y : -x ** 2 - (y - 4) ** 2 + 4
print(min(z)) # -48


fig, axis = plt.subplots(1, 1, figsize=(14, 10))
# gridsize=150
gridsize=3 # needs to < N, which the discretization steps of one axis
nx=50
ny=2
# im = axis.hexbin(x, y, C=z, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=-0.9, vmax=2.1)
# im = axis.hexbin(x, y, C=z, gridsize=(nx,nx-2), cmap=cm.jet, bins=None)
im = axis.hexbin(x, y, C=z, gridsize=nx, cmap=cm.jet, bins=None, vmin=-48., vmax=4.)
axis.axis([x.min(), x.max(), y.min(), y.max()])
axis.set_aspect('equal')

cb = fig.colorbar(im, )
cb.set_label('Value')
plt.show()