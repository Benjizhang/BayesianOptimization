# script to read the exp log and replay the exp in simulation mode
# 
# Z. Zhang
# 2022/08

from pathlib import Path
import pandas as pd
import numpy as np
import os, os.path, glob
import matplotlib.pyplot as plt
from boa_helper import plot_2d2,plot_2d3,plot_2d4,SfatyPara
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.gaussian_process.kernels import RBF,Matern

## exp parameters
sp = SfatyPara()
originx = sp.originX
originy = sp.originY
originz = sp.originZ

## log path
expFolderName = '/202208081435_UR_BOA_spiral' # <<<<<<
NutStorePath = 'D:/02data/MyNutFiles/我的坚果云'
dataPath = NutStorePath+expFolderName+'/data'
figPath = NutStorePath+expFolderName+'/fig'


## cal the num of slides
num_data_files = len([name for name in os.listdir(dataPath) if os.path.isfile(os.path.join(dataPath, name))])
if num_data_files % 3 == 0:
    num_slides = num_data_files//3
else: raise Exception('[ERR] The Num of Data is Incorrect!')

## read xxx_slidex_BOA.csv from the given list 'slide_id_ls'
os.chdir(dataPath)
slide_id_ls = range(1,num_slides+1)
boa_data_ls = [None] * len(slide_id_ls)
boa_s_g_ls = [None] * len(slide_id_ls)
for i in range(len(slide_id_ls)):
    slide_id = slide_id_ls[i]
    file_suffix = '*slide'+str(slide_id)+'_BOA.csv'
    # file_suffix = '*_BOA.csv'
    ## find the specific file
    for file in glob.glob(file_suffix):
        print('---- {} ----'.format(file))
        boa_data_ls[i] = pd.read_csv(file,header=0,names=['ite','posx','posy','boaVal'])
        boa_s_g_ls[i] = pd.read_csv(file,nrows=1,names=['sx','sy','gx','gy'])
        # print(boa_s_g_ls[i])

## plot the BOA pts with coloers according to values
xmin = 0.0
xmax = 0.25
ymin = 0.0
ymax = 0.35
vmin=0
vmax=7
# for cur_boa in boa_data_ls:
# for k in range(len(slide_id_ls)):
#     cur_boa = boa_data_ls[k]
#     cur_sg  = boa_s_g_ls[k]
#     # print(cur_boa)
#     # print(cur_boa.posx)
#     pt_nonjamming = cur_boa[cur_boa.boaVal <=5]
#     pt_object     = cur_boa[cur_boa.boaVal >=5]
#     # print(pt_nonjamming.head())
#     # print(pt_object.head())
#     plt.plot(pt_nonjamming.posx,pt_nonjamming.posy,'D', markersize=1, color='k')
#     plt.plot(pt_object.posx,pt_object.posy,'D', markersize=1, color = 'r')
#     plt.plot(cur_sg.sx,cur_sg.sy,'x', markersize=5, color = 'k') # start pt
#     plt.plot(cur_sg.gx,cur_sg.gy,'*', markersize=5, color = 'k') # goal
#     plt.axis('scaled')
#     plt.axis([xmin, xmax, ymin, ymax])
#     plt.show()

## replay the exp proces

## ------ BOA ------ ##
## BOA init. (bounds in the relatvie frame)
##--- BOA related codes ---#
# kernel = RBF(length_scale=8, length_scale_bounds='fixed')
# kernel = Matern(length_scale=1, length_scale_bounds='fixed',nu=np.inf)
lenScaleBound ='fixed'
# lenScaleBound = (1e-5, 1e5)
# lenScaleBound = (0.01, 0.2)
kernel = Matern(length_scale=0.1, length_scale_bounds=lenScaleBound, nu=np.inf)
# kernel = Matern(length_scale=0.04, length_scale_bounds=lenScaleBound, nu=np.inf)
# kernel = Matern(length_scale=0.04, length_scale_bounds=lenScaleBound, nu=2.5)
# kernel = Matern(length_scale=0.04, length_scale_bounds=lenScaleBound, nu=1.5)
str_kernel = str(kernel)
## initial distribution in BOA
xrange = np.linspace(sp.xmin, sp.xmax, sp.xBoaSteps)
yrange = np.linspace(sp.ymin, sp.ymax, sp.yBoaSteps)
X, Y = np.meshgrid(xrange, yrange)
xrange = X.ravel()
yrange = Y.ravel()
XY = np.vstack([xrange, yrange]).T
## buried object shape
obj1 = np.array([[0.035,0.175],
[0.035,0.255],
[0.115,0.175]])
obj2 = np.array([[0.155,0.085],
[0.207,0.141],
[0.237,0.116],
[0.185,0.06]])
object_shape = [obj1, obj2]

bo = BayesianOptimization(f=None, pbounds={'x': (sp.xmin, sp.xmax), 'y': (sp.ymin, sp.ymax)},
                    verbose=2,
                    random_state=1)
# plt.ioff()
bo.set_gp_params(kernel=kernel)
util = UtilityFunction(kind="ei", 
                    kappa = 2, 
                    xi=0.5,
                    kappa_decay=1,
                    kappa_decay_delay=0)

## replay the exp proces
f_max=7
for k in range(len(slide_id_ls)):
    slide_id = slide_id_ls[k]
    cur_boa = boa_data_ls[k]
    cur_sg  = boa_s_g_ls[k]
    cur_boa = cur_boa.sort_values(by=['ite'],ascending=True) # ascending
    # pt_nonjamming = cur_boa[cur_boa.boaVal <=5]
    # pt_object     = cur_boa[cur_boa.boaVal >=5]
    # plt.plot(pt_nonjamming.posx,pt_nonjamming.posy,'D', markersize=1, color='k')
    # plt.plot(pt_object.posx,pt_object.posy,'D', markersize=1, color = 'r')
    # plt.plot(cur_sg.sx,cur_sg.sy,'x', markersize=5, color = 'k') # start pt
    # plt.plot(cur_sg.gx,cur_sg.gy,'*', markersize=5, color = 'k') # goal
    # plt.axis('scaled')
    # plt.axis([xmin, xmax, ymin, ymax])
    # plt.show()
    for index, row in cur_boa.iterrows():
        print(row['posx'], row['posy'])
        boax = row['posx']
        boay = row['posy']
        if row['boaVal'] <=5:
            fd = 3.0
        else: 
            fd = 7.0
        probePt_dict = {'x':boax,'y':boay}                        
        bo.register(params=probePt_dict, target=fd)

    
    for index, row in cur_sg.iterrows():
        probeCurPt = [row['gx'],row['gy']]
    ## BOA plot 2d (v4)    
    # plot_2d4(slide_id, bo, util, kernel, xrange,yrange,XY, f_max, 'dd', probeCurPt, object_shape, "{:03}".format(len(bo._space.params)))



