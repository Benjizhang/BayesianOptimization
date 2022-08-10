# some supporting func for BOA

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
# from functions.saftyCheck import SfatyPara

# class includes the important sfaty hard constraints
class SfatyPara:

    def __init__(self):
        
        # safety force threshold
        self.FORCE_MAX  = 15.   # 15 N
        # safety penetration force
        self.PENE_FORCE_MAX  = 11.   # 15 N

        # lift/penetration limits
        self.LIFT_Z_MIN = +0.08 # +8 cm
        self.LIFT_Z_MAX = +0.20 # +20 cm
        self.PENE_Z_MIN = -0.   # -0 cm
        self.PENE_Z_MAX = -0.06 # -6 cm

        # origin coordinates in the UR base frame
        self.originX = -0.5931848696000094
        self.originY = -0.28895797651231064
        self.originZ = 0.07731254732208744
        # safe box (relative frame)        
        self.xmin = 0.   # 0 cm
        self.xmax = 0.25 # +25 cm
        self.ymin = 0.   # 0 cm
        self.ymax = 0.35 # +25 cm
        self.zmin = self.PENE_Z_MAX # -6 cm
        self.zmax = self.LIFT_Z_MAX # +20cm
        # safe box (UR base frame)        
        self.XMIN = self.originX + self.xmin
        self.XMAX = self.originX + self.xmax
        self.YMIN = self.originY + self.ymin
        self.YMAX = self.originY + self.ymax
        self.ZMIN = self.originZ + self.zmin
        self.ZMAX = self.originZ + self.zmax
        # safety height to translation
        self.SAFEZ = self.originZ + 0.10 # +10cm

        # discretization steps for BOA plot (related to the xmin/xmax/ymin/ymax)
        self.xBoaSteps = 125
        self.yBoaSteps = 175
    
    # check a 3d position is in the limit or not
    def checkCoorLimit3d(self,pos):
        curx = pos[0]
        cury = pos[1]
        curz = pos[2]
        if curx >= self.XMIN and curx <= self.XMAX and \
           cury >= self.YMIN and cury <= self.YMAX and \
           curz >= self.ZMIN and curz <= self.ZMAX:
            return True
        else:
            raise Exception("ERROR: Out of Range")

    # check whether in the X-Y limit 
    def checkCoorLimitXY(self,pos):
        curx = pos[0]
        cury = pos[1]
        if curx >= self.XMIN and curx <= self.XMAX and \
            cury >= self.YMIN and cury <= self.YMAX:
            return True
        else: 
            raise Exception("ERROR: Out of Range")

### plot_2d (v2)
def plot_2d2(slide_id, bo, util, kernel,x,y,XY, f_max, fig_path, name=None):

    mu, s, ut = posterior(bo, util, XY)
    fig, ax = plt.subplots(2, 2, figsize=(9, 8))
    gridsize=88
    str_kernel = str(kernel)
    
    fig.suptitle('{}-th slide, {} {}'.format(slide_id,str_kernel,kernel.length_scale_bounds), fontdict={'size':30})
    
    # GP regression output
    ax[0][0].set_title('GP Predicted Mean', fontdict={'size':15})
    im00 = ax[0][0].hexbin(x, y, C=mu, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=f_max)
    ax[0][0].axis('scaled')
    ax[0][0].axis([x.min(), x.max(), y.min(), y.max()])
    ax[0][0].plot(bo._space.params[:, 0], bo._space.params[:, 1], 'D', markersize=1, color='k', label='Observations')
    
    ax[0][1].set_title('Target Function', fontdict={'size':15})
    im01 = ax[0][1].hexbin(x, y, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=1)
    ax[0][1].axis('scaled')
    ax[0][1].axis([x.min(), x.max(), y.min(), y.max()])

    ax[1][0].set_title('GP Variance', fontdict={'size':15})
    im10 = ax[1][0].hexbin(x, y, C=s, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=1)
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
    

    for im, axis in zip([im00, im01, im10, im11], ax.flatten()):
        cb = fig.colorbar(im, ax=axis)
        # cb.set_label('Value')

    if name is None:
        name = '_'

    # plt.tight_layout()

    ## Save or show figure?
    fig.savefig(fig_path + name + '.png')
    # plt.ioff()
    # plt.show()
    # plt.pause(2)
    # plt.close(fig)

### plot_2d (v3)
## plot cur pos and next goal
def plot_2d3(slide_id, bo, util, kernel,x,y,XY, f_max, fig_path, curPt, name=None):
    sp = SfatyPara()
    mu, s, ut = posterior(bo, util, XY)
    fig, ax = plt.subplots(2, 2, figsize=(9, 8))
    gridsize=88
    str_kernel = str(kernel)
    
    fig.suptitle('{}-th slide, {} {}'.format(slide_id,str_kernel,kernel.length_scale_bounds), fontdict={'size':30})
    
    # GP regression output
    ax[0][0].set_title('GP Predicted Mean', fontdict={'size':15})
    im00 = ax[0][0].hexbin(x, y, C=mu, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=f_max)
    ax[0][0].axis('scaled')
    ax[0][0].axis([x.min(), x.max(), y.min(), y.max()])
    ax[0][0].plot(bo._space.params[:, 0], bo._space.params[:, 1], 'D', markersize=1, color='k', label='Observations')
    ## plot cur pos
    ax[0][0].plot(curPt[0], curPt[1], 'x', markersize=5, color='k')
    ## plot the next target
    # ax[0][0].plot(np.where(ut.reshape((175, 125)) == ut.max())[1]*0.25/125.,
    #             np.where(ut.reshape((175, 125)) == ut.max())[0]*0.35/175.,
    #             '*', markersize=5, color='k')
    if round(ut.max()-ut.min(),6) != 0:
        ax[0][0].plot(np.where(ut.reshape((sp.yBoaSteps, sp.xBoaSteps)) == ut.max())[1]*(sp.xmax - sp.xmin)/sp.xBoaSteps,
                    np.where(ut.reshape((sp.yBoaSteps, sp.xBoaSteps)) == ut.max())[0]*(sp.ymax - sp.ymin)/sp.yBoaSteps,
                    '*', markersize=5, color='k')

    ax[0][1].set_title('Target Function', fontdict={'size':15})
    im01 = ax[0][1].hexbin(x, y, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=1)
    ax[0][1].axis('scaled')
    ax[0][1].axis([x.min(), x.max(), y.min(), y.max()])

    ax[1][0].set_title('GP Variance', fontdict={'size':15})
    im10 = ax[1][0].hexbin(x, y, C=s, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=1)
    ax[1][0].axis('scaled')
    ax[1][0].axis([x.min(), x.max(), y.min(), y.max()])    
    # plt.pause(0.1)

    ax[1][1].set_title('Acquisition Function', fontdict={'size':15})    
    im11 = ax[1][1].hexbin(x, y, C=ut, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=min(ut), vmax=max(ut))
    ax[1][1].axis('scaled')
    ax[1][1].axis([x.min(), x.max(), y.min(), y.max()])
    # plt.pause(0.1)    

    for im, axis in zip([im00, im01, im10, im11], ax.flatten()):
        cb = fig.colorbar(im, ax=axis)
        # cb.set_label('Value')

    if name is None:
        name = '_'

    ## Save or show figure?
    fig.savefig(fig_path + name + '.png')
    # plt.ioff()
    # plt.show()
    # plt.pause(2)
    # plt.close(fig)

### plot_2d (v4)
## plot buried object
def plot_2d4(slide_id, bo, util, kernel,x,y,XY, f_max, fig_path, curPt, object_shape, name=None):
    sp = SfatyPara()
    mu, s, ut = posterior(bo, util, XY)
    fig, ax = plt.subplots(2, 2, figsize=(9, 8))
    gridsize=88
    str_kernel = str(kernel)
    
    fig.suptitle('{}-th slide, {} {}'.format(slide_id,str_kernel,kernel.length_scale_bounds), fontdict={'size':30})
    
    # GP regression output
    ax[0][0].set_title('GP Predicted Mean', fontdict={'size':15})
    im00 = ax[0][0].hexbin(x, y, C=mu, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=f_max)
    ax[0][0].axis('scaled')
    ax[0][0].axis([x.min(), x.max(), y.min(), y.max()])
    ax[0][0].plot(bo._space.params[:, 0], bo._space.params[:, 1], 'D', markersize=1, color='k', label='Observations')
    ## plot cur pos
    ax[0][0].plot(curPt[0], curPt[1], 'x', markersize=5, color='k')
    ## plot the next target
    # ax[0][0].plot(np.where(ut.reshape((175, 125)) == ut.max())[1]*0.25/125.,
    #             np.where(ut.reshape((175, 125)) == ut.max())[0]*0.35/175.,
    #             '*', markersize=5, color='k')
    if round(ut.max()-ut.min(),6) != 0:
        ax[0][0].plot(np.where(ut.reshape((sp.yBoaSteps, sp.xBoaSteps)) == ut.max())[1]*(sp.xmax - sp.xmin)/sp.xBoaSteps,
                    np.where(ut.reshape((sp.yBoaSteps, sp.xBoaSteps)) == ut.max())[0]*(sp.ymax - sp.ymin)/sp.yBoaSteps,
                    '*', markersize=5, color='k')
    ## plot buried object shape
    num_obj = len(object_shape)
    for i in range(num_obj):
        cur_object = object_shape[i]
        cur_object_x = np.hstack((cur_object[:,0],cur_object[0,0]))
        cur_object_y = np.hstack((cur_object[:,1],cur_object[0,1]))
        ax[0][0].plot(cur_object_x,cur_object_y,'k-')

    ax[0][1].set_title('Target Function', fontdict={'size':15})
    im01 = ax[0][1].hexbin(x, y, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=1)
    ax[0][1].axis('scaled')
    ax[0][1].axis([x.min(), x.max(), y.min(), y.max()])

    ax[1][0].set_title('GP Variance', fontdict={'size':15})
    im10 = ax[1][0].hexbin(x, y, C=s, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=0, vmax=1)
    ax[1][0].axis('scaled')
    ax[1][0].axis([x.min(), x.max(), y.min(), y.max()])    
    # plt.pause(0.1)

    ax[1][1].set_title('Acquisition Function', fontdict={'size':15})    
    im11 = ax[1][1].hexbin(x, y, C=ut, gridsize=gridsize, cmap=cm.jet, bins=None, vmin=min(ut), vmax=max(ut))
    ax[1][1].axis('scaled')
    ax[1][1].axis([x.min(), x.max(), y.min(), y.max()])
    # plt.pause(0.1)    

    for im, axis in zip([im00, im01, im10, im11], ax.flatten()):
        cb = fig.colorbar(im, ax=axis)
        # cb.set_label('Value')

    if name is None:
        name = '_'

    ## Save or show figure?
    # fig.savefig(fig_path + name + '.png')
    # plt.ioff()
    plt.show()
    # plt.pause(2)
    # plt.close(fig)


### posterior
def posterior(bo, util, X):
    ur = unique_rows(bo._space.params)
    bo._gp.fit(bo._space.params[ur], bo._space.target[ur])
    mu, sigma2 = bo._gp.predict(X, return_std=True)
    ac = util.utility(X, bo._gp, bo._space.target.max())

    return mu, np.sqrt(sigma2), ac

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