# function to smooth the drag force with vibration
# 
# Z Zhang
# 07/2022

import random
import numpy
import pylab 

def smooth_drag_force(A, H, f_range, f_read, d_c = 0.01):
    g = 9.8067 # m/s^2
    f_theory = A*g*d_c*(H**2)
    if abs(f_read - f_theory) <= f_range:
        state = 'non-jamming'
        f_smooth = f_theory
    else:
        state = 'jamming'
        f_smooth = f_read

    return state, f_smooth

## using Kalman Filter to smooth the drag force (Fd)
def smooth_fd_kf(fd_read_ls):
    ## parameters for KF
    n_iter = len(fd_read_ls)
    sz = (n_iter,) 
    z = fd_read_ls
    Q = 1e-5   # process variance (state func)
    R = 0.1**2 # estimate of measurement variance (measure func)

    ## assign memory 
    xhat=numpy.zeros(sz)      # x 滤波估计值  
    P=numpy.zeros(sz)         # 滤波估计协方差矩阵  
    xhatminus=numpy.zeros(sz) #  x 估计值  
    Pminus=numpy.zeros(sz)    # 估计协方差矩阵  
    K=numpy.zeros(sz)         # 卡尔曼增益

    ## intial guesses  
    xhat[0] = 0.0  
    P[0] = 1.0 

    ## KF iterations 
    for k in range(1,n_iter): 
        # predict  
        xhatminus[k] = xhat[k-1]  #X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0  
        Pminus[k] = P[k-1]+Q      #P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1  
        
        # update  
        K[k] = Pminus[k]/( Pminus[k]+R ) #Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1  
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k]) #X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1  
        P[k] = (1-K[k])*Pminus[k] #P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1  

    return xhat, Pminus



if __name__ == '__main__':
    A = 10728. # kg/m^3
    g = 9.8067 # m/s^2 
    dc = 0.01 # m 
    depth = 0.05 # m
    f_the = A*g*dc*(depth**2)
    
    ## using 'smooth_drag_force'
    # f_rand = f_the + random.uniform(-1.5,1.5)
    # print(f_rand)
    # s_out, f_out = smooth_drag_force(A,depth,1,f_rand,0.01)
    # print(s_out, f_out)

    ## using 'smooth_fd_kf'
    n_iter = 50  
    sz = (n_iter,)
    fd_read_ls = numpy.random.normal(f_the,1,size=sz)
    fdhat, Pminus = smooth_fd_kf(fd_read_ls)
    pylab.figure()  
    pylab.plot(fd_read_ls,'k+',label='noisy measurements')     #观测值  
    pylab.plot(fdhat,'b-',label='a posteri estimate')  #滤波估计值  
    pylab.axhline(f_the,color='g',label='truth value')    #真实值  
    pylab.legend()  
    pylab.xlabel('Iteration')  
    pylab.ylabel('Force (N)')  

    pylab.figure()  
    valid_iter = range(1,n_iter) # Pminus not valid at step 0  
    pylab.plot(valid_iter,Pminus[valid_iter],label='a priori error estimate')  
    pylab.xlabel('Iteration')  
    pylab.ylabel('$(Force)^2$')  
    pylab.setp(pylab.gca(),'ylim',[0,.01])  
    pylab.show()

