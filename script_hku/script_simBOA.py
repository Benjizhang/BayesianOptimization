# script to simulate BOA-guided pre-touch exploration strategy using GRAINS
# (based on script_BoTest\script_simulator_spiral.py)
# ---- sudo codes ----
# for-loop:
#   1- tell goal pt by BOA
#   2- GRAINS detection:
#    2.1 No object found after traj. then tell BOA 0 in raked area
#    2.2 Find object: 
#        1) tell BOA 0 at stop position
#        2) probe: pull up -> move 1cm forward -> penetrate into GM:
#            a. can penetrate: report false positive -> keep moving along spiral traj.
#            b. cannot pentrate: tell BOA 1 at this pos -> repeat 2) step 
# end of for-loop
#
# Z Zhang
# 05/2023

import numpy as np
from bayes_opt import BayesianOptimization, UtilityFunction
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from sklearn.gaussian_process.kernels import RBF,Matern
# from smooth_f import smooth_drag_force


# kernel = RBF(length_scale=8, length_scale_bounds='fixed')
# kernel = Matern(length_scale=1, length_scale_bounds='fixed',nu=np.inf)
lenScaleBound ='fixed'
# lenScaleBound = (1e-5, 1e5)
# lenScaleBound = (0.01, 0.02)
kernel = Matern(length_scale=0.1, length_scale_bounds=lenScaleBound, nu=np.inf)
# kernel = Matern(length_scale=0.04, length_scale_bounds=lenScaleBound, nu=np.inf)
# kernel = Matern(length_scale=0.04, length_scale_bounds=lenScaleBound, nu=2.5)
# kernel = Matern(length_scale=0.04, length_scale_bounds=lenScaleBound, nu=1.5)
str_kernel = str(kernel)

##--- BOA related codes ---##

### unique_rows
def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]

### posterior
def posterior(bo, X):
    ur = unique_rows(bo._space.params)
    bo._gp.fit(bo._space.params[ur], bo._space.target[ur])
    mu, sigma2 = bo._gp.predict(X, return_std=True)
    ac = util.utility(X, bo._gp, bo._space.target.max())

    return mu, np.sqrt(sigma2), ac

### plot_2d_flip
def plot_2d_flip(bo, XY, f_max, name=None):
    mu, s, ut = posterior(bo, XY)
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    gridsize=88

    plt.ion()

    # GP regression output
    ax[0][0].set_title('Gausian Process Predicted Mean', fontdict={'size':15})
    im00 = ax[0][0].hexbin(y, x, C=mu, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=f_max)
    # ax[0][0].axis([x.min(), x.max(), y.min(), y.max()])
    # ax[0][0].plot(bo._space.params[:, 0], bo._space.params[:, 1], 'D', markersize=4, color='k', label='Observations')
    # ax[0][0].plot(xbd,ybd,'k-', lw=2, color='k')
    ax[0][0].axis([y.min(), y.max(), x.min(), x.max()])
    ax[0][0].plot(bo._space.params[:, 1], bo._space.params[:, 0], 'D', markersize=4, color='k', label='Observations')
    ## convert x,y label
    ax[0][0].set_ylim(ax[0][0].get_ylim()[::-1]) 
    ax[0][0].xaxis.tick_top()
    ax[0][0].yaxis.tick_left()    
    ax[0][0].set_xlabel('y')    
    ax[0][0].xaxis.set_label_position('top')
    ax[0][0].set_ylabel("x")

    ax[0][1].set_title('Target Function', fontdict={'size':15})
    im10 = ax[0][1].hexbin(y, x, C=z, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=f_max)
    # ax[0][1].axis([x.min(), x.max(), y.min(), y.max()])
    # ax[0][1].plot(bo._space.params[:, 0], bo._space.params[:, 1], 'D', markersize=4, color='k')
    # ax[0][1].plot(xbd,ybd,'k-', lw=2, color='k')
    ax[0][1].axis([y.min(), y.max(), x.min(), x.max()])
    ax[0][1].plot(bo._space.params[:, 1], bo._space.params[:, 0], 'D', markersize=4, color='k')
    ## convert x,y label
    ax[0][1].set_ylim(ax[0][1].get_ylim()[::-1]) 
    ax[0][1].xaxis.tick_top()
    ax[0][1].yaxis.tick_left()    
    ax[0][1].set_xlabel('y')    
    ax[0][1].xaxis.set_label_position('top')
    ax[0][1].set_ylabel("x")


    ax[1][0].set_title('Gausian Process Variance', fontdict={'size':15})
    # im01 = ax[1][0].hexbin(x, y, C=s, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=1)
    # ax[1][0].axis([x.min(), x.max(), y.min(), y.max()])
    im01 = ax[1][0].hexbin(y, x, C=s, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=1)
    ax[1][0].axis([y.min(), y.max(), x.min(), x.max()])
    ## convert x,y label
    ax[1][0].set_ylim(ax[1][0].get_ylim()[::-1]) 
    ax[1][0].xaxis.tick_top()
    ax[1][0].yaxis.tick_left()    
    ax[1][0].set_xlabel('y')    
    ax[1][0].xaxis.set_label_position('top')
    ax[1][0].set_ylabel("x")

    ax[1][1].set_title('Acquisition Function', fontdict={'size':15})
    # im11 = ax[1][1].hexbin(x, y, C=ut, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=8)
    im11 = ax[1][1].hexbin(y, x, C=ut, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=8)

    print(ut)
    maxVal_x = np.where(ut.reshape((125, 175)) == ut.max())[0]
    maxVal_y = np.where(ut.reshape((125, 175)) == ut.max())[1]
    print(np.where(ut.reshape((125, 175)) == ut.max()))
    print(maxVal_x)
    print(maxVal_y)

    ax[1][1].plot([np.where(ut.reshape((125, 175)) == ut.max())[1]*0.35/175.,
                   np.where(ut.reshape((125, 175)) == ut.max())[1]*0.35/175.],
                  [0, 0.25],
                  '-', lw=2, color='k')
    # plt.show()

    ax[1][1].plot([0, 0.35],
                  [np.where(ut.reshape((125, 175)) == ut.max())[0]*0.25/125.,
                   np.where(ut.reshape((125, 175)) == ut.max())[0]*0.25/125.],                  
                  '-', lw=2, color='k')
    # plt.show()
    
    ax[1][1].axis([y.min(), y.max(), x.min(), x.max()])
    ## convert x,y label
    ax[1][1].set_ylim(ax[1][1].get_ylim()[::-1]) 
    ax[1][1].xaxis.tick_top()
    ax[1][1].yaxis.tick_left()    
    ax[1][1].set_xlabel('y')    
    ax[1][1].xaxis.set_label_position('top')
    ax[1][1].set_ylabel("x")

    

    for im, axis in zip([im00, im10, im01, im11], ax.flatten()):
        cb = fig.colorbar(im, ax=axis)
        # cb.set_label('Value')

    if name is None:
        name = '_'

    plt.tight_layout()
    plt.axis('equal')

    # Save or show figure?
    # fig.savefig('./figures/fourLine/'+'boa_eg_' + name + '.png')
    # plt.show()
    plt.close(fig)

### plot_2d
def plot_2d(ite, bo, XY, f_max, f_sigma, name=None):

    mu, s, ut = posterior(bo, XY)
    #self._space.params, self._space.target
    fig, ax = plt.subplots(2, 2, figsize=(9, 8))
    gridsize=88
    
    fig.suptitle('Slide: {}-th ite, fsigma {} N, {} {}'.format(ite,f_sigma,str_kernel,kernel.length_scale_bounds), fontdict={'size':30})
    
    # GP regression output
    ax[0][0].set_title('GP Predicted Mean', fontdict={'size':15})
    im00 = ax[0][0].hexbin(x, y, C=mu, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=f_max)
    ax[0][0].axis('scaled')
    ax[0][0].axis([x.min(), x.max(), y.min(), y.max()])
    ax[0][0].plot(bo._space.params[:, 0], bo._space.params[:, 1], 'D', markersize=1, color='k', label='Observations')
    # ax[0][0].plot(xbd,ybd,'k-', lw=2, color='k')    
    # plt.pause(0.1)

    ax[0][1].set_title('Target Function', fontdict={'size':15})
    im10 = ax[0][1].hexbin(x, y, C=z, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=f_max)
    ax[0][1].axis('scaled')
    ax[0][1].axis([x.min(), x.max(), y.min(), y.max()])
    # ax[0][1].plot(bo._space.params[:, 0], bo._space.params[:, 1], 'D', markersize=4, color='k')
    # print(bo._space.target)
    # len(bo._space.target)
    # ax[0][1].plot(xbd,ybd,'k-', lw=2, color='k')    
    # plt.pause(0.1)

    ax[1][0].set_title('GP Variance', fontdict={'size':15})
    im01 = ax[1][0].hexbin(x, y, C=s, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=1)
    ax[1][0].axis('scaled')
    ax[1][0].axis([x.min(), x.max(), y.min(), y.max()])    
    # plt.pause(0.1)

    ax[1][1].set_title('Acquisition Function', fontdict={'size':15})    
    im11 = ax[1][1].hexbin(x, y, C=ut, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=min(ut), vmax=max(ut))
    ax[1][1].axis('scaled')
    ax[1][1].axis([x.min(), x.max(), y.min(), y.max()])
    # plt.pause(0.1)

    # region
    # print(ut)
    # maxVal_x = np.where(ut.reshape((175, 125)) == ut.max())[0]
    # maxVal_y = np.where(ut.reshape((175, 125)) == ut.max())[1]
    # print(np.where(ut.reshape((175, 125)) == ut.max()))
    # print(maxVal_x)
    # print(maxVal_y)

    # ax[1][1].plot([np.where(ut.reshape((175, 125)) == ut.max())[1]*0.25/125.,
    #                np.where(ut.reshape((175, 125)) == ut.max())[1]*0.25/125.],
    #               [0, 0.35],
    #               '-', lw=2, color='k')
    # plt.show()

    # ax[1][1].plot([0, 0.25],
    #               [np.where(ut.reshape((175, 125)) == ut.max())[0]*0.35/175.,
    #                np.where(ut.reshape((175, 125)) == ut.max())[0]*0.35/175.],
    #               '-', lw=2, color='k')
    # plt.show()
    # endregion
    

    for im, axis in zip([im00, im10, im01, im11], ax.flatten()):
        cb = fig.colorbar(im, ax=axis)
        # cb.set_label('Value')

    if name is None:
        name = '_'

    # plt.tight_layout()

    ## Save or show figure?
    # fig.savefig('./figures/GMSim/'+'boa_eg_' + name + '.png')
    plt.ioff()
    # plt.show()
    plt.pause(2)
    # plt.close(fig)
    
def plot_2d_wObj(ite, bo, XY, f_max, f_sigma, name=None):

    mu, s, ut = posterior(bo, XY)
    #self._space.params, self._space.target
    fig, ax = plt.subplots(2, 2, figsize=(9, 8))
    gridsize=88
    
    fig.suptitle('Slide: {}-th ite, fsigma {} N, {} {}'.format(ite,f_sigma,str_kernel,kernel.length_scale_bounds), fontdict={'size':30})
    
    # GP regression output
    ax[0][0].set_title('GP Predicted Mean', fontdict={'size':15})
    im00 = ax[0][0].hexbin(x, y, C=mu, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=f_max)
    ax[0][0].axis('scaled')
    ax[0][0].axis([x.min(), x.max(), y.min(), y.max()])
    ax[0][0].plot(bo._space.params[:, 0], bo._space.params[:, 1], '.', markersize=1, color='k', label='Observations')
    ax[0][0].plot([0.05,0.1,0.1,0.05,0.05],[0.15,0.15,0.2,0.2,0.15],'k-')
    ## plot cur pos
    ax[0][0].plot(bo._space.params[-1, 0], bo._space.params[-1, 1], 'x', markersize=5, color='k')
    ## plot the next target
    ax[0][0].plot(np.where(ut.reshape((175, 125)) == ut.max())[1]*0.25/125.,
                np.where(ut.reshape((175, 125)) == ut.max())[0]*0.35/175.,
                '*', markersize=5, color='k')
    # ax[0][0].plot(xbd,ybd,'k-', lw=2, color='k')
    # plt.pause(0.1)

    ax[0][1].set_title('Target Function', fontdict={'size':15})
    im10 = ax[0][1].hexbin(x, y, C=z, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=f_max)
    ax[0][1].axis('scaled')
    ax[0][1].axis([x.min(), x.max(), y.min(), y.max()])
    # ax[0][1].plot(bo._space.params[:, 0], bo._space.params[:, 1], 'D', markersize=4, color='k')
    # print(bo._space.target)
    # len(bo._space.target)
    # ax[0][1].plot(xbd,ybd,'k-', lw=2, color='k')    
    # plt.pause(0.1)

    ax[1][0].set_title('GP Variance', fontdict={'size':15})
    im01 = ax[1][0].hexbin(x, y, C=s, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=1)
    ax[1][0].axis('scaled')
    ax[1][0].axis([x.min(), x.max(), y.min(), y.max()])    
    # plt.pause(0.1)

    ax[1][1].set_title('Acquisition Function', fontdict={'size':15})    
    im11 = ax[1][1].hexbin(x, y, C=ut, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=min(ut), vmax=max(ut))
    ax[1][1].axis('scaled')
    ax[1][1].axis([x.min(), x.max(), y.min(), y.max()])
    # plt.pause(0.1)    

    for im, axis in zip([im00, im10, im01, im11], ax.flatten()):
        cb = fig.colorbar(im, ax=axis)
        # cb.set_label('Value')

    if name is None:
        name = '_'

    # plt.tight_layout()

    ## Save or show figure?
    # fig.savefig('./figures/GMSim/'+'boa_eg_' + name + '.png')
    plt.ioff()
    # plt.show()
    plt.pause(2)
    # plt.close(fig)

### target
def target(x, y, H=0.03):
    A = 10728. # kg/m^3
    g = 9.8067 # m/s^2
    dc = 0.01 # m    
    F_sigma = 1 # N
    Fd = 0.*x + 0.*y + A*g*dc*(H**2) + F_sigma * random.uniform(-1,1) # N

    return Fd

def dragF_exact(x,y,H=0.03):
    A = 10728. # kg/m^3
    g = 9.8067 # m/s^2
    dc = 0.01 # m    
    # F_sigma = 1 # N
    # Fd = 0.*x + 0.*y + A*g*dc*(H**2) + F_sigma * random.uniform(-1,1) # N

    centx = 0.2
    centy = 0.2
    f_mean = A*g*dc*(H**2)
    raise_coeff = 2.35
    f_max = raise_coeff * f_mean
    d_range = 0.04
    d_cur = np.sqrt((x-centx)**2+(y-centy)**2)
    if round(d_cur,6) <= d_range:
        coeff = (f_mean - f_max)/(d_range**2)
        Fd = coeff*(d_cur**2) + f_max
    else:
        Fd = f_mean
    
    return Fd

def dragF_noise(x,y,H=0.03):
    A = 10728. # kg/m^3
    g = 9.8067 # m/s^2
    dc = 0.01 # m    
    F_sigma = 0.2 # N
    # Fd = 0.*x + 0.*y + A*g*dc*(H**2) + F_sigma * random.uniform(-1,1) # N

    centx = 0.1
    centy = 0.15
    f_mean = A*g*dc*(H**2)
    raise_coeff = 2.35
    f_max = raise_coeff * f_mean
    d_range = 0.04
    d_cur = np.sqrt((x-centx)**2+(y-centy)**2)
    F_noise = F_sigma * random.uniform(-1,1) # N
    # print(random.uniform(-1,1))
    if round(d_cur,6) <= d_range:
        coeff = (f_mean - f_max)/(d_range**2)
        Fd = coeff*(d_cur**2) + f_max + F_noise
    else:
        Fd = f_mean + F_noise
    
    return Fd

def dragF_noise2(x,y,H=0.03,F_sigma=0.2):
    A = 10728. # kg/m^3
    g = 9.8067 # m/s^2
    dc = 0.01 # m    
    # N
    # Fd = 0.*x + 0.*y + A*g*dc*(H**2) + F_sigma * random.uniform(-1,1) # N

    centx = 0.1
    centy = 0.15
    f_mean = A*g*dc*(H**2)
    raise_coeff = 2.5
    f_max = raise_coeff * f_mean
    d_range = 0.04
    d_cur = np.sqrt((x-centx)**2+(y-centy)**2)
    F_noise = F_sigma * random.uniform(-1,1) # N
    # print(random.uniform(-1,1))
    if round(d_cur,6) <= d_range:
        coeff = (f_mean - f_max)/(d_range**2)
        Fd = coeff*(d_cur**2) + f_max + F_noise
    else:
        Fd = f_mean + F_noise
    
    return Fd

def dragF_noise3(x,y,H=0.03,F_sigma=0.2):
    ## change the model of jamming
    A = 10728. # kg/m^3
    g = 9.8067 # m/s^2
    dc = 0.01 # m 

    centx = 0.1
    centy = 0.15
    f_mean = A*g*dc*(H**2)
    raise_coeff = 2.5
    f_max = raise_coeff * f_mean
    d_range = 0.04
    d_cur = np.sqrt((x-centx)**2+(y-centy)**2)
    F_noise = F_sigma * random.uniform(-1,1) # N
    # print(random.uniform(-1,1))
    if round(d_cur,6) <= d_range:
        cc = f_max
        bb = 2*(f_mean - f_max)/d_range
        aa = -(f_mean - f_max)/(d_range**2)
        Fd = aa*(d_cur**2) + bb * d_cur + cc + F_noise
    else:
        Fd = f_mean + F_noise
    
    return Fd

### inObj
def inObj(ptx,pty):
    seg_a = lambda x,y : x+y-7
    seg_b = lambda x,y : -x-y+5
    seg_c = lambda x,y : -x+y-2
    seg_d = lambda x,y : x-y-1
    if np.sign(seg_a(ptx,pty)) < 0 and np.sign(seg_b(ptx,pty)) < 0 and np.sign(seg_c(ptx,pty)) < 0 and np.sign(seg_d(ptx,pty)) < 0:
        return True
    else:
        return False


### gen_slide_path2
def gen_slide_path2(endPt, startPt={'x':0,'y':0}, d_c = 0.1):
    ex = list(endPt.values())[0]
    ey = list(endPt.values())[1]
    sx = list(startPt.values())[0]
    sy = list(startPt.values())[1]
    path_len = np.sqrt((ex-sx)**2+(ey-sy)**2)
    num_pt = int((path_len//d_c))
    intptx = []
    intpty = []
    for i in range(num_pt+1):
        s = i / (num_pt)
        curPtx = sx + s*(ex - sx)
        curPty = sy + s*(ey - sy)        
        intptx.append(curPtx)
        intpty.append(curPty)
    
    return intptx, intpty

### gen_slide_path3 (including start and end pts)
def gen_slide_path3(endPt, startPt={'x':0,'y':0}, d_c = 0.01):
    ex = list(endPt.values())[0]
    ey = list(endPt.values())[1]
    sx = list(startPt.values())[0]
    sy = list(startPt.values())[1]
    path_len = np.sqrt((ex-sx)**2+(ey-sy)**2)
    num_pt = int((path_len//d_c))
    intptx = []
    intpty = []
    sinAng = (ey - sy)/path_len
    cosAng = (ex - sx)/path_len
    # including start pt (i from 0)
    for i in range(0,num_pt+1):
        curPtx = sx + i * d_c * cosAng
        curPty = sy + i * d_c * sinAng        
        intptx.append(curPtx)
        intpty.append(curPty)
    # including end pt
    intptx.append(ex)
    intpty.append(ey)

    return intptx, intpty

def plotInitSandBox_flip(x,y,z):
    # # move Origin to upper left corner
    
    ## initial distribution
    # x = np.linspace(0, xp, 125)
    # y = np.linspace(0, yp, 175)
    # X, Y = np.meshgrid(x, y)
    # x = X.ravel()
    # y = Y.ravel()
    # XY = np.vstack([x, y]).T
    # z = 0*x

    # plt.ion()
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
    # plt.ylabel("x")
    axis.set_ylabel('x')

    cb = fig.colorbar(im, )
    cb.set_label('Value')
    plt.show()

def plotInitSandBox(x,y,z):
    ## normal layout
    # plt.ion()
    fig, axis = plt.subplots(1, 1)
    axis.axis('scaled')
    gridsize=88
    im = axis.hexbin(x, y, C=z, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=7)
    # plt.text(0.1,0.22,'star')
    xmin, xmax, ymin, ymax = axis.axis([x.min(), x.max(), y.min(), y.max()])
    plt.ylabel("y")
    plt.xlabel("x")

    cb = fig.colorbar(im, )
    cb.set_label('Value')
    plt.pause(0.1)

################################################################
if __name__ == '__main__':
    SAFE_FORCE = 15.0
    xp = 0.25
    xn = 0.0
    yp = 0.35
    yn = 0.0

    ## BOA init.
    bo = BayesianOptimization(f=None, pbounds={'x': (0, xp), 'y': (0, yp)},
                        verbose=2,
                        random_state=1)
    plt.ioff()
    bo.set_gp_params(kernel=kernel)
    util = UtilityFunction(kind="ei", 
                        kappa = 2, 
                        xi=0.5,
                        kappa_decay=1,
                        kappa_decay_delay=0)
    curPt = {'x':0.1,'y':0}
    ## initial distribution in BOA
    x = np.linspace(0, xp, 125)
    y = np.linspace(0, yp, 175)
    X, Y = np.meshgrid(x, y)
    x = X.ravel()
    y = Y.ravel()
    XY = np.vstack([x, y]).T
    # z = 0*x
    zls = []
    # random.seed(233)
    f_sigma = 1
    for i in range(len(x)):
        curx = x[i]
        cury = y[i]
        zls.append(dragF_noise3(curx,cury,0.05,f_sigma))
    z = np.array(zls)
    ## plot the init distribution
    # plotInitSandBox(x,y,np.array(zls))

    ## if given seed, goals would be identical for each run
    # random.seed(2)

    ## probe slides in the granular media
    goalx = [0]                   
    goaly = [0]
    # random.seed(233)
    plotPath = 1
    A = 10728.
    fig_path = plt.figure()
    ax_path = fig_path.gca()
    for k in range(1,61):
        print("--------- {}-th slide ---------".format(k))
        ######### cal. goal by BOA #########        
        # BOA provides the relative goal
        nextPt = bo.suggest(util) # dict type    
        ex = list(nextPt.values())[0]
        ey = list(nextPt.values())[1]
        goalx.append(ex)
        goaly.append(ey)
        print('relative goal x {:.3f}, y {:.3f}'.format(ex,ey))
        ### penetrate into the goals ###
        # probePtz = dragF_noise3(ex,ey,0.05,f_sigma)
        # bo.register(params=nextPt, target=probePtz)
        
        #region: drag force models
        ## [shceme 1] using 'smooth_drag_force'
        ### slides to the goal ###
        # intptx, intpty = gen_slide_path3(endPt=nextPt, startPt=curPt, d_c = 0.01)
        # for i in range(len(intptx))[1:]:
        #     ## form the point in dict. type
        #     probePt_dict = {'x':intptx[i],'y':intpty[i]}
        #     probePtz = dragF_noise3(intptx[i],intpty[i],0.05,f_sigma)
        #     ##------- main contribution -------##
        #     ## smooth the drag forces measured from FT300s
        #     s_out, f_out = smooth_drag_force(A,0.05,1,probePtz,0.01)            
        #     print('Cur Pos x {:.3f}, y {:.3f}, s {}'.format(intptx[i],intpty[i],s_out))
        #     print('Drag   force: {:.3f} N'.format(probePtz))
        #     print('Smooth force: {:.3f} N'.format(f_out))
        #     bo.register(params=probePt_dict, target=f_out)     
        #     if s_out == 'jamming':
        #         print('===== jamming =====')
        #     ##======= main contribution =======##       
        
        ## [shceme 2] 
        ### using collected data (slide1-slide15)
        # data = [[1020,0.1003,0.0099,3],
        #     # [2263,0.1006,0.0199,3],
        #     # [3499,0.1008,0.0299,3],
        #     [4735,0.1011,0.0399,3],
        #     # [5967,0.1013,0.0499,3],
        #     # [7199,0.1015,0.0599,3],
        #     [8430,0.1017,0.0699,3],
        #     [9564,0.1065,0.0776,7.0032],
        #     [980,0.1084,0.0871,7.062],
        #     [382,0.1064,0.0834,7.004],
        #     [952,0.1093,0.0924,7.2265],
        #     [363,0.1069,0.0893,7.0628],
        #     [863,0.1129,0.0959,7.0981],
        #     [413,0.1115,0.0915,7.1347],
        #     [845,0.118,0.0973,7.0131],
        #     [868,0.1239,0.1039,7.1997],
        #     [256,0.1213,0.1036,7.3713],
        #     [965,0.1237,0.113,7.0655],[594,0.1286,0.1089,7.0397],[910,0.1333,0.1168,7.0449],[461,0.1332,0.1118,7.0343],[1022,0.133,0.1218,7.1733],
        #     ]
        # for row in data:
        #     print(row)
        #     ## form the point in dict. type
        #     probePt_dict = {'x':row[1],'y':row[2]}
        #     f_out = row[3]
        #     bo.register(params=probePt_dict, target=f_out) 
        #     plot_2d(k, bo, XY, 10, 0, "{:03}".format(len(bo._space.params)))

        ## [shceme 3] 
        # ### cannot get into the object
        # intptx, intpty = gen_slide_path3(endPt=nextPt, startPt=curPt, d_c = 0.01)
        # obj_xmin = 0.05
        # obj_xmax = 0.1
        # obj_ymin = 0.15
        # obj_ymax = 0.2
        # obj_cent = [(obj_xmin+obj_xmax)/2,(obj_ymin+obj_ymax)/2]
        # obj_range = np.sqrt((obj_xmax-obj_xmin)**2+(obj_ymax-obj_ymin)**2)*0.8
        # for i in range(len(intptx))[1:]:
        #     if not (obj_xmin <= intptx[i] and intptx[i] <= obj_xmax and obj_ymin <= intpty[i] and intpty[i]<=obj_ymax):
        #         probePt_dict = {'x':intptx[i],'y':intpty[i]}
        #         # not in the obj.
        #         dist = np.sqrt((intptx[i] - obj_cent[0])**2+(intpty[i] - obj_cent[1])**2)
        #         if dist >= obj_range:
        #             fd = 3
        #         else:
        #             fd = 7 
        #         bo.register(params=probePt_dict, target=fd) 
        # plot_2d_wObj(k, bo, XY, 10, 0, "{:03}".format(len(bo._space.params)))
        
        ## [shceme 4] 
        # ### cannot get into the object (virtually give fd max in the range of obj)
        # intptx, intpty = gen_slide_path3(endPt=nextPt, startPt=curPt, d_c = 0.01)
        # obj_xmin = 0.05
        # obj_xmax = 0.1
        # obj_ymin = 0.15
        # obj_ymax = 0.2
        # obj_cent = [(obj_xmin+obj_xmax)/2,(obj_ymin+obj_ymax)/2]
        # obj_range = np.sqrt((obj_xmax-obj_xmin)**2+(obj_ymax-obj_ymin)**2)*0.8
        # for i in range(len(intptx))[1:]:            
        #     probePt_dict = {'x':intptx[i],'y':intpty[i]}
        #     # not in the obj.
        #     dist = np.sqrt((intptx[i] - obj_cent[0])**2+(intpty[i] - obj_cent[1])**2)
        #     if dist >= obj_range:
        #         fd = 3
        #     else:
        #         fd = 7 
        #     bo.register(params=probePt_dict, target=fd) 
        # # plot_2d_wObj(k, bo, XY, 10, 0, "{:03}".format(len(bo._space.params)))
        # endregion

        ## [shceme 5] fd = 7N for pts in the object; fd = 3N for pts in the object
        ### cannot get into the object (virtually give fd max in the obj)
        intptx, intpty = gen_slide_path3(endPt=nextPt, startPt=curPt, d_c = 0.01)
        obj_xmin = 0.05
        obj_xmax = 0.1
        obj_ymin = 0.15
        obj_ymax = 0.2
        obj_cent = [(obj_xmin+obj_xmax)/2,(obj_ymin+obj_ymax)/2]
        obj_range = np.sqrt((obj_xmax-obj_xmin)**2+(obj_ymax-obj_ymin)**2)*0.8
        obj_margin = 0.02
        for i in range(len(intptx))[1:]:
            if not (obj_xmin-obj_margin<= intptx[i] and intptx[i] <= obj_xmax+obj_margin and obj_ymin-obj_margin <= intpty[i] and intpty[i]<=obj_ymax+obj_margin):
                # not in the obj.
                fd = 3
            else:
                fd = 7 
            probePt_dict = {'x':intptx[i],'y':intpty[i]}
            bo.register(params=probePt_dict, target=fd) 
        # if k >=10:
            # plot_2d_wObj(k, bo, XY, 10, 0, "{:03}".format(len(bo._space.params)))

        if plotPath == 1:
            # plotInitSandBox(x,y,np.array(zls))
            ## plot the sliding path
            ax_path.plot(goalx,goaly,'k-') 
            ax_path.axis('scaled')
            ax_path.axis([0, 0.25, 0,0.35])    
            plt.pause(1)
        
        # probe goes to the nextPt
        curPt = {'x':ex,'y':ey}
        if k >=10 and k%10 ==0:
        # # if k >=1:
            # plot_2d(k, bo, XY, 7, f_sigma, "{:03}".format(len(bo._space.params)))
            plot_2d_wObj(k, bo, XY, 10, 0, "{:03}".format(len(bo._space.params)))

    print('shut down')