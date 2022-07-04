# test the drag force model
# 
# Z Zhang
# 07/2022

import random
import numpy as np
import script_simulate_slide_noise as ssn
import matplotlib.pyplot as plt


d_c = 0.001
curPt = {'x':0,'y':0}
nextPt = {'x':0.25,'y':0.3}
intptx, intpty = ssn.gen_slide_path3(endPt=nextPt, startPt=curPt, d_c=d_c)
# print(intptx,intpty)

## plot the individual points on the path
fig, axis = plt.subplots(1, 1, figsize=(14, 10))
plt.plot(intptx,intpty,'.') 
plt.axis('scaled')
plt.axis([0, 0.25, 0,0.35])    
# plt.pause(5)
plt.show()

f_sigma =1
f_ls = []
## measure the drag forces
for i in range(len(intptx))[1:]:
    probePt_dict = {'x':intptx[i],'y':intpty[i]}
    probePtz = ssn.dragF_noise3(intptx[i],intpty[i],0.05,f_sigma)          
    f_ls.append(probePtz)  
    print('Cur Pos x {:.3f}, y {:.3f}'.format(intptx[i],intpty[i]))
    print('Drag force: {:.3f} N'.format(probePtz))

fig, axis = plt.subplots(1, 1, figsize=(14, 10))
fig.suptitle('d\_c: {}, fsigma {} N'.format(d_c,f_sigma), fontdict={'size':30})


plt.plot(np.dot(np.arange(i),d_c),f_ls,'-')
# plt.axis('scaled')
plt.axis([0,np.dot(np.arange(i),d_c).max(),0,10])   
plt.xlabel('m') 
# plt.pause(5)
# plt.ioff()
plt.grid()
plt.show()

