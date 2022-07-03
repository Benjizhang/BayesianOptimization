# plot the initial distribution (0 N) of sand box
# move Origin to upper left corner
# 
# Z Zhang
# 06/2022

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

xp = 0.25
xn = 0.0
yp = 0.35
yn = 0.0
SAFE_FORCE = 5

x = np.linspace(0, xp, 125)
y = np.linspace(0, yp, 175)
X, Y = np.meshgrid(x, y)
x = X.ravel()
y = Y.ravel()
XY = np.vstack([x, y]).T
z = 0*x

plt.ion()
fig, axis = plt.subplots(1, 1)
axis.axis('scaled')
gridsize=88
im = axis.hexbin(y, x, C=z, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=SAFE_FORCE)
# plt.text(0.1,0.22,'star')
xmin, xmax, ymin, ymax = axis.axis([y.min(), y.max(), x.min(), x.max()])
axis.set_ylim(axis.get_ylim()[::-1]) 
axis.xaxis.tick_top()
axis.yaxis.tick_left()    
axis.set_xlabel('y')    
axis.xaxis.set_label_position('top')   
# plt.xlabel("y")
axis.set_ylabel("x")


cb = fig.colorbar(im, )
cb.set_label('Value')
plt.show()

## example：move Origin to upper left corner

plt.figure()    
plt.axis([0, 16, 0, 16])     
plt.grid(False)                         # set the grid
data= [(8,2,'USA'), (2,8,'CHN')]

for obj in data:
    # plt.text(obj[1],obj[0],obj[2])      # change x,y as there is no view() in mpl
    plt.text(obj[0],obj[1],obj[2]) 

ax=plt.gca()                            # get the axis
ax.set_ylim(ax.get_ylim()[::-1])        # invert the axis
ax.xaxis.tick_top()                     # and move the X-Axis      
ax.yaxis.set_ticks(np.arange(0, 16, 1)) # set y-ticks
# ax.yaxis.tick_left()