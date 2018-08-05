#!/usr/bin/env python3

# The following code is developed by Sisi Li 
# Please contact sisli@umich.edu if you have any questions

#linear interpolation

from scipy.interpolate import interp1d
import numpy as np
paths = np.load('np_array_path.npy')
interpolation_test = []
for id_paths in range (1, 40):
    id_paths_detail = paths[:,[4,5,7,8]][paths[:,1] == id_paths]
    id_paths_detail = id_paths_detail[0:int(len(id_paths_detail)/2)]
    interpolation_test.append(list(id_paths_detail))



for id_path in range (1,20):
    if len(interpolation_test[id_path]) < 200:
        inter = np.array(interpolation_test[id_path])
        inter_col0 = inter[:,0]
        inter_col1 = inter[:,1]
        inter_col2 = inter[:,2]
        inter_col3 = inter[:,3]
        #inter_col4 = inter[:,4]
        x_0 = np.linspace(inter_col0[0],inter_col0[-1],num = len(inter_col0),endpoint = True)
        x_1 = np.linspace(inter_col1[0],inter_col1[-1],num = len(inter_col0),endpoint = True)
        x_2 = np.linspace(inter_col2[0],inter_col2[-1],num = len(inter_col0),endpoint = True)
        x_3 = np.linspace(inter_col3[0],inter_col3[-1],num = len(inter_col0),endpoint = True)
        #x_4 = np.linspace(inter_col4[0],inter_col4[-1],num = len(inter_col0),endpoint = True)
        y_0 = inter_col0
        y_1 = inter_col1
        y_2 = inter_col2
        y_3 = inter_col3
       # y_4 = inter_col4
        f_0 = interp1d(x_0,y_0)
        f_1 = interp1d(x_1,y_1)
        f_2 = interp1d(x_2,y_2)
        f_3 = interp1d(x_3,y_3)
        #f_4 = interp1d(x_4,y_4)
        xnew_0 = np.linspace(inter_col0[0],inter_col0[-1],num = 200, endpoint = True)
        xnew_1 = np.linspace(inter_col1[0],inter_col1[-1],num = 200, endpoint = True)
        xnew_2 = np.linspace(inter_col2[0],inter_col2[-1],num = 200, endpoint = True)
        xnew_3 = np.linspace(inter_col3[0],inter_col3[-1],num = 200, endpoint = True)
        #xnew_4 = np.linspace(inter_col4[0],inter_col4[-1],num = 200, endpoint = True)
        ynew_0 = f_0(xnew_0)
        ynew_1 = f_1(xnew_1)
        ynew_2 = f_2(xnew_2)
        ynew_3 = f_3(xnew_3)
        new_inter = np.zeros((200,4))
        new_inter[:,0] = ynew_0
        new_inter[:,1] = ynew_1
        new_inter[:,2] = ynew_2
        new_inter[:,3] = ynew_3
        
import matplotlib.pyplot as plt
plt.plot(x_1, y_1, 'o', xnew_1, f_1(xnew_1), '-')
#plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()       

  
        
        
        
    

    

