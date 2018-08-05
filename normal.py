#!/usr/bin/env python3

# The following code is developed by Sisi Li
# Please contact sisli@umich.edu if you have any questions

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from scipy.interpolate import interp1d 

#Importing the dataset 
#paths = pd.read_csv('vehicle_Encountering_detail.csv')
#paths = np.array(paths)
data = np.load('np_array_path.npy')
# data process

training_data = []
for id_paths in range (1, 40):
    id_paths_detail = data[:,[4,5,7,8]][data[:,1] == id_paths]
    id_paths_detail = id_paths_detail[0:int(len(id_paths_detail)/2)]
    training_data.append(list(id_paths_detail))
   
#add speed
temp_total= []
for i in range(len(training_data)):
    temp_paths = training_data[i]  # list 
    temp = []    
    for m in range(len(temp_paths) - 1):
        speed_A = math.sqrt((temp_paths[m + 1][0] - temp_paths[m][0])**2 + 
                       (temp_paths[m + 1][1] - temp_paths[m][1])**2)
        speed_B = math.sqrt((temp_paths[m + 1][2] - temp_paths[m][2])**2 + 
                       (temp_paths[m + 1][3] - temp_paths[m][3])**2)

        speed_relative = math.fabs(speed_A - speed_B)
        temp_= np.append(temp_paths[m],speed_relative)
        temp.append(list(temp_))
        
    temp_total.append(list(temp))   
training_data = temp_total

   # np.append(np.array([1,2,3]),2)
   # interpolation
for id_path in range (len(training_data)):
   if len(training_data[id_path]) < 200:
        inter = np.array(training_data[id_path])
        inter_col0 = inter[:,0]
        inter_col1 = inter[:,1]
        inter_col2 = inter[:,2]
        inter_col3 = inter[:,3]
        inter_col4 = inter[:,4]
        x_0 = np.linspace(inter_col0[0],inter_col0[-1],num = len(inter_col0),endpoint = True)
        x_1 = np.linspace(inter_col1[0],inter_col1[-1],num = len(inter_col0),endpoint = True)
        x_2 = np.linspace(inter_col2[0],inter_col2[-1],num = len(inter_col0),endpoint = True)
        x_3 = np.linspace(inter_col3[0],inter_col3[-1],num = len(inter_col0),endpoint = True)
        x_4 = np.linspace(inter_col4[0],inter_col4[-1],num = len(inter_col0),endpoint = True)
        y_0 = inter_col0
        y_1 = inter_col1
        y_2 = inter_col2
        y_3 = inter_col3
        y_4 = inter_col4
        f_0 = interp1d(x_0,y_0)
        f_1 = interp1d(x_1,y_1)
        f_2 = interp1d(x_2,y_2)
        f_3 = interp1d(x_3,y_3)
        f_4 = interp1d(x_4,y_4)
        
        xnew_0 = np.linspace(inter_col0[0],inter_col0[-1],num = 200, endpoint = True)
        xnew_1 = np.linspace(inter_col1[0],inter_col1[-1],num = 200, endpoint = True)
        xnew_2 = np.linspace(inter_col2[0],inter_col2[-1],num = 200, endpoint = True)
        xnew_3 = np.linspace(inter_col3[0],inter_col3[-1],num = 200, endpoint = True)
        xnew_4 = np.linspace(inter_col4[0],inter_col4[-1],num = 200, endpoint = True)
        ynew_0 = f_0(xnew_0)
        ynew_1 = f_1(xnew_1)
        ynew_2 = f_2(xnew_2)
        ynew_3 = f_3(xnew_3)
        ynew_4 = f_4(xnew_4)
        new_inter = np.zeros((200,5))
        new_inter[:,0] = ynew_0
        new_inter[:,1] = ynew_1
        new_inter[:,2] = ynew_2
        new_inter[:,3] = ynew_3
        new_inter[:,4] = ynew_4
        training_data[id_path] = list(new_inter)
    
   
#Normal
#training_data[i]: carA latitude, carA longitude, carB latitude, carB longitude, velocity difference
Normal_trainning_set = []
v = []
for i in range(len(training_data)):
    sc = MinMaxScaler(feature_range = (0, 1))
    length = len(training_data[i])
    carLatitudes = np.zeros([1,2 * length])
    carLongitudes = np.zeros([1, 2 * length])
    #velocities = np.zeros([1, length])
    temp = np.array(training_data[i])
    carLatitudes[0,0:length] = temp[:,0]
    carLatitudes[0,length: 2*length] = temp[:,2]
    carLongitudes[0,0:length] = temp[:,1]
    carLongitudes[0,length: 2*length] = temp[:,3]
#        for k in range(length):
#            carLatitudes[0, k] = training_data[i][k][0]
#            carLatitudes[0, k + length] = training_data[i][k][2]
#            carLongitudes[0, k] = training_data[i][k][1]
#            carLongitudes[0, k + length] = training_data[i][k][3]
#            velocities[0, k] = training_data[i][k][4]       
    carLatitudes = sc.fit_transform(np.transpose(carLatitudes))
    carLongitudes = sc.fit_transform(np.transpose(carLongitudes))
    temp[:,0] = carLatitudes[0:length,0]
    temp[:,2] = carLatitudes[length: 2*length,0]
    temp[:,1] = carLongitudes[0:length,0]
    temp[:,3] = carLongitudes[length: 2*length,0]
    #velocities = sc.fit_transform(velocities)  
#        Normal = np.zeros((length, 5))
#        for k in range(length):
#            Normal[k, 0] = carLatitudes[0, k]
#            Normal[k, 1] = carLongitudes[0, k]
#            Normal[k, 2] = carLatitudes[0, k + length]
#            Normal[k, 3] = carLongitudes[0, k + length]
#            Normal[k, 4] = velocities[0, k] 
               
    Normal_trainning_set.append(list(temp))
    v.append(list(temp[:,4]))


max_v = max(max(v))
min_v = min(min(v))

for i in range(len(training_data)):
    for j in range(len(training_data[i])):
        Normal_trainning_set[i][j][4] =  Normal_trainning_set[i][j][4]/max_v
       
        




#Sample
Sample_trainning_set = []
for i in range(len(Normal_trainning_set)):
    array_size = len(Normal_trainning_set[i])
    step = int(array_size/200)
    if step > 0:
        Normal = np.array(Normal_trainning_set[i])
        sample = Normal[0: 200*step: step]
        Sample_trainning_set.append(list(sample))
  

