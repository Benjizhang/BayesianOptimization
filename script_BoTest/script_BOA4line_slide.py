# 4-line (#-shaped) boundary distribution
# slide the probe
# 

from xml.etree.ElementTree import PI
from bayes_opt import BayesianOptimization, UtilityFunction
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import mlab
from matplotlib import gridspec
import copy
from collections import Iterable

def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    # print(a)
    # print(a.T)
    order = np.lexsort(a.T)
    # print(order)
    reorder = np.argsort(order)
    # print(reorder)

    a = a[order]
    # print(a)
    diff = np.diff(a, axis=0)
    # print(diff)
    ui = np.ones(len(a), 'bool')
    # print(ui)
    ui[1:] = (diff != 0).any(axis=1)
    # print(ui)

    return ui[reorder]

def ptsRectangle(RectCent,RectOrie,L,W):
    ptsls = [[L/2,W/2],[-L/2,W/2],[-L/2,-W/2],[L/2,-W/2]]
    # cent = np.array(RectCent)
    orie_rad = np.deg2rad(RectOrie)
    rotMat = [[np.cos(orie_rad), -np.sin(orie_rad)],
            [np.sin(orie_rad),  np.cos(orie_rad)]]
    ptxy_glb = []
    for i in range(4):
        onePt = (ptsls)[i]
        ptxy_glb.append(list(RectCent + np.matmul(rotMat,onePt)))
    return ptxy_glb


def target2(x, y):
    a = np.exp(-( (x - 2)**2/0.7 + (y - 4)**2/1.2) + (x - 2)*(y - 4)/1.6 )
    b = np.exp(-( (x - 4)**2/3 + (y - 2)**2/2.) )
    c = np.exp(-( (x - 4)**2/0.5 + (y - 4)**2/0.5) + (x - 4)*(y - 4)/0.5 )
    d = np.sin(3.1415 * x)
    e = np.exp(-( (x - 5.5)**2/0.5 + (y - 5.5)**2/.5) )
    # return 2*a + b - c + 0.17 * d + 2*e
    return a + b

def target2circle(x, y):
    a = np.exp(-(x - 1) ** 2 - (y - 3) ** 2 + 1)
    b = np.exp(-(x - 4) ** 2 - (y - 2) ** 2 + 1.6)
    return a+b

# rectangle distribution
def target(x, y):
    a = np.exp(x+y-7)
    b = np.exp(-x-y+5)
    c = np.exp(-x+y-2)
    d = np.exp(x-y-1)
    return 4-(a+b+c+d)

n = 1e5
x = y = np.linspace(0, 6, 300)
X, Y = np.meshgrid(x, y)
x = X.ravel()
y = Y.ravel()
# print(x)
# print(np.vstack([x, y]))
# X = np.vstack([x, y]).T[:, [1, 0]]
X = np.vstack([x, y]).T
z = target(x, y)

zmin = -5
zmax = 4
# zmin = min(z)
# zmax = max(z)
print(min(z))
print(max(z))




fig, axis = plt.subplots(1, 1, figsize=(14, 10))
gridsize=150

# im = axis.hexbin(x, y, C=z, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=-0.9, vmax=2.1)
im = axis.hexbin(x, y, C=z, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=zmin, vmax=zmax)
# im = axis.hexbin(x, y, C=z, gridsize=gridsize, cmap=cm.jet, bins='log', vmin=zmin, vmax=zmax)
axis.axis([x.min(), x.max(), y.min(), y.max()])

cb = fig.colorbar(im, )
cb.set_label('Value')
plt.show()

util = UtilityFunction(kind='ucb',
                       kappa=10,
                       xi=0.0,
                       kappa_decay=1,
                       kappa_decay_delay=0)


def posterior(bo, X):
    ur = unique_rows(bo._space.params)
    bo._gp.fit(bo._space.params[ur], bo._space.target[ur])
    mu, sigma2 = bo._gp.predict(X, return_std=True)
    ac = util.utility(X, bo._gp, bo._space.target.max())
    return mu, np.sqrt(sigma2), ac



def plot_2d(name=None):

    mu, s, ut = posterior(bo, X)
    #self._space.params, self._space.target
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    gridsize=150

    # fig.suptitle('Bayesian Optimization in Action', fontdict={'size':30})

    # GP regression output
    ax[0][0].set_title('Gausian Process Predicted Mean', fontdict={'size':15})
    im00 = ax[0][0].hexbin(x, y, C=mu, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=zmin, vmax=zmax)
    ax[0][0].axis([x.min(), x.max(), y.min(), y.max()])
    # ax[0][0].plot(bo._space.params[:, 1], bo._space.params[:, 0], 'D', markersize=4, color='k', label='Observations')
    ax[0][0].plot(bo._space.params[:, 0], bo._space.params[:, 1], 'D', markersize=4, color='k', label='Observations')

    ax[0][1].set_title('Target Function', fontdict={'size':15})
    im10 = ax[0][1].hexbin(x, y, C=z, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=zmin, vmax=zmax)
    ax[0][1].axis([x.min(), x.max(), y.min(), y.max()])
    # ax[0][1].plot(bo._space.params[:, 1], bo._space.params[:, 0], 'D', markersize=4, color='k')
    ax[0][1].plot(bo._space.params[:, 0], bo._space.params[:, 1], 'D', markersize=4, color='k')


    ax[1][0].set_title('Gausian Process Variance', fontdict={'size':15})
    im01 = ax[1][0].hexbin(x, y, C=s, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=1)
    ax[1][0].axis([x.min(), x.max(), y.min(), y.max()])

    ax[1][1].set_title('Acquisition Function', fontdict={'size':15})
    im11 = ax[1][1].hexbin(x, y, C=ut, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=8)
    print(ut)
    maxVal_x = np.where(ut.reshape((300, 300)) == ut.max())[0]
    maxVal_y = np.where(ut.reshape((300, 300)) == ut.max())[1]
    print(np.where(ut.reshape((300, 300)) == ut.max()))
    print(maxVal_x)
    print(maxVal_y)

    ax[1][1].plot([np.where(ut.reshape((300, 300)) == ut.max())[1]/50.,
                   np.where(ut.reshape((300, 300)) == ut.max())[1]/50.],
                  [0, 6],
                  'k-', lw=2, color='k')
    # plt.show()

    ax[1][1].plot([0, 6],
                  [np.where(ut.reshape((300, 300)) == ut.max())[0]/50.,
                   np.where(ut.reshape((300, 300)) == ut.max())[0]/50.],
                  'k-', lw=2, color='k')
    # plt.show()

    ax[1][1].axis([x.min(), x.max(), y.min(), y.max()])

    for im, axis in zip([im00, im10, im01, im11], ax.flatten()):
        cb = fig.colorbar(im, ax=axis)
        # cb.set_label('Value')

    if name is None:
        name = '_'

    plt.tight_layout()

    # Save or show figure?
    fig.savefig('./figures/fourLine/'+'boa_eg_' + name + '.png')
    # plt.show()
    # plt.pause(3)
    # plt.close(fig)

def gen_slide_path(endPt, startPt={'x':0,'y':0}, num_pt = 11):
    ex = list(endPt.values())[0]
    ey = list(endPt.values())[1]
    sx = list(startPt.values())[0]
    sy = list(startPt.values())[1]
    intptx = []
    intpty = []
    for i in range(num_pt):
        s = i / (num_pt - 1)
        intptx.append(sx + s*(ex - sx))
        intpty.append(sy + s*(ey - sy))
    return intptx, intpty





bo = BayesianOptimization(target, {'x': (0, 6), 'y': (0, 6)})
plt.ioff()
# -------------- slide --------------
util = UtilityFunction(kind="ei", 
                    kappa = 2, 
                    xi=0.5,
                    kappa_decay=1,
                    kappa_decay_delay=0)
curPt = {'x':0,'y':0}
for i in range(50):
    nextPt = bo.suggest(util) # dict type
    # generate the slide segment
    intptx, intpty = gen_slide_path(endPt=nextPt, startPt=curPt, num_pt = 6)
    print(intptx)
    print(intpty)
    # probe at these points (excluding start pt)
    for i in range(len(intptx))[1:]:
        # form the point in dict. type
        probePt_dict = {'x':intptx[i],'y':intpty[i]}
        probePtz = target(**probePt_dict)
        bo.register(params=probePt_dict, target=probePtz)
    
    # probe goes to the nextPt
    curPt = copy.deepcopy(nextPt)
    plot_2d("{:03}".format(len(bo._space.params)))
# ============== slide ==============

# bo.maximize(init_points=5, n_iter=0, acq='ucb', kappa=10)
# plot_2d("{:03}".format(len(bo._space.params)))



# Turn interactive plotting off
# plt.ioff()
# plt.ion()

# for i in range(50):
    # bo.maximize(init_points=0, n_iter=1, acq='ucb', kappa=10)
    # plot_2d("{:03}".format(len(bo._space.params)))
