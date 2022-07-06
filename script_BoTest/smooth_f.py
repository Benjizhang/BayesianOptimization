# function to smooth the drag force with vibration
# 
# Z Zhang
# 07/2022

import random

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

if __name__ == '__main__':
    A = 10728. # kg/m^3
    g = 9.8067 # m/s^2 
    dc = 0.01 # m 
    depth = 0.05 # m
    f_the = A*g*dc*(depth**2)
    f_rand = f_the + random.uniform(-1.5,1.5)
    print(f_rand)
    s_out, f_out = smooth_drag_force(A,depth,1,f_rand,0.01)
    print(s_out, f_out)
