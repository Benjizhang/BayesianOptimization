def black_box_function(x, y):
    """Function with unknown internals we wish to maximize.

    This is just serving as an example, for all intents and
    purposes think of the internals of this function, i.e.: the process
    which generates its output values, as unknown.
    """
    return -x ** 2 - (y - 1) ** 2 + 1

import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization

fig = plt.figure()
ax = plt.axes(projection='3d')

np.random.seed(42)
xs = np.linspace(2, 4, 100)
ys = np.linspace(-3, 3, 100)
## line 
# ax.plot3D(xs,ys,zs)

X, Y = np.meshgrid(xs, ys)
Z = black_box_function(X,Y)
## contour
# ax.set_title('3D contour')
# ax.contour3D(X, Y, Z, 50, cmap='binary')

## wireframe
# ax.set_title('wireframe')
# ax.plot_wireframe(X, Y, Z, color='black')

## Surface
ax.set_title('Surface plot')
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

# Bounded region of parameter space
pbounds = {'x': (2, 4), 'y': (-3, 3)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=1,
)

optimizer.maximize(
    init_points=5,
    n_iter=5,
)