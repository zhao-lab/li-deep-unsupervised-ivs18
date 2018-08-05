#!/usr/bin/env python3

# The following code is developed by Sisi Li 
# Please contact sisli@umich.edu if you have any questions

# AutoEncoders

# Importing the libraries
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
paths = np.load('np_array_path.npy')


# data process
def convert(data):
    training_data = []
    for id_paths in range (1, 4000):
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
    print('Converting finished!')
    for i in range(len(training_data)):
        sc = MinMaxScaler(feature_range = (0, 1))
        length = len(training_data[i])
        carLatitudes = np.zeros([1,2 * length])
        carLongitudes = np.zeros([1, 2 * length])
        #velocities = np.zeros([1, length])
        temp = np.array(training_data[i])
        carLatitudes[:,0:length] = temp[:,0]
        carLatitudes[:,length: 2*length] = temp[:,2]
        carLongitudes[:,0:length] = temp[:,1]
        carLongitudes[:,length: 2*length] = temp[:,3]
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
        v.append(list(temp[:,4]))            
        Normal_trainning_set.append(list(temp))
    
    # Normal_speed
    max_v = max(max(v))
    min_v = min(min(v))

    for i in range(len(Normal_trainning_set)):
        for j in range(len(Normal_trainning_set[i])):
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
      
    return Sample_trainning_set

trainning_set = convert(paths)
    
#Converting the data into Torch tensors

training_set = torch.FloatTensor(trainning_set)
nb_paths = len(trainning_set)
nb_paths_length = 200
# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_paths_length, 50)
        self.fc2 = nn.Linear(50, 25)
        self.fc3 = nn.Linear(25, 50)
        self.fc4 = nn.Linear(50, nb_paths_length)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x1 = self.activation(self.fc1(x))
        x2 = self.activation(self.fc2(x1))
        x3 = self.activation(self.fc3(x2))
        x4 = self.activation(self.fc4(x3))
        return x2,x4
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)

# Training the SAE
nb_epoch = 300
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_path in range(nb_paths):
        input = Variable(torch.t(training_set[id_path]))
        target = input.clone()
        output = sae(input)[1]
        target.require_grad = False
        loss = criterion(output, target)
        loss.backward()
        train_loss += np.sqrt(loss.data[0])
        s += 1.
        optimizer.step()
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    
    
#k_means cluster the feature from autoencoder
    
Kmeans_data= []
for id_path in range(nb_paths):
    input = Variable(torch.t(training_set[id_path]))
    kmeans_data = sae(input)[0]
    kmeans_data = kmeans_data.data.numpy()
    kmeans_data = kmeans_data.reshape(125)
    Kmeans_data.append(kmeans_data)
    
from sklearn.cluster import KMeans 

cluster_nb = 40

kmeans = KMeans(n_clusters = cluster_nb, random_state = 0).fit(Kmeans_data)

import os
import os.path
import shutil
for n in range(cluster_nb):
    if os.path.exists("label_{}".format(n)):
        shutil.rmtree("label_{}".format(n))
    os.mkdir("label_{}".format(n))
    
    
for n in range(cluster_nb):
    if os.path.exists("label_3D_{}".format(n)):
        shutil.rmtree("label_3D_{}".format(n))
    os.mkdir("label_3D_{}".format(n))
    

for n in range(cluster_nb):
    if os.path.exists("label_speed_{}".format(n)):
        shutil.rmtree("label_speed_{}".format(n))
    os.mkdir("label_speed_{}".format(n))
    


for id_path in range(nb_paths):
    graph = np.array(trainning_set[id_path])
    plt.plot(graph[0,0],graph[0,1],color = 'blue',marker = 'o',markersize = 7)   
    plt.plot(graph[:,0], graph[:,1], 'bo', markersize= 1)  
    plt.plot(graph[199,0],graph[199,1],color = 'blue',marker = '+',markersize = 7)
    
    plt.plot(graph[0,2],graph[0,3],color = 'red',marker = 'o',markersize = 7)  
    plt.plot(graph[:,2], graph[:,3], 'ro',markersize = 1)
    plt.plot(graph[199,2],graph[199,3], color = 'red',marker = '+', markersize = 7)  
     
    label = kmeans.labels_[id_path]
    plt.savefig("./label_{}/{}.jpg".format(label, id_path),dpi = 600)  
    plt.clf()
 
for id_path in range(nb_paths):
    graph = np.array(trainning_set[id_path])
    plt.plot(graph[:,4], 'bo', markersize= 1)     
    label = kmeans.labels_[id_path]
    plt.savefig("./label_speed_{}/{}_speed.jpg".format(label, id_path),dpi = 600)  
    plt.clf()



from mpl_toolkits.mplot3d import Axes3D  
for id_path in range(nb_paths):
    graph = np.array(trainning_set[id_path])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z = np.linspace(0, 10, 200)
    ax.plot(graph[:,0], graph[:,1], z,'bo',markersize = 1)  
    ax.plot(graph[:,2], graph[:,3], z,'ro',markersize = 1)   
    label = kmeans.labels_[id_path]
    plt.savefig("./label_3D_{}/{}.jpg".format(label, id_path),dpi = 600)  
    plt.clf()
    

#for id_path in range(nb_paths):
#    data = np.array(trainning_set[id_path])
#    label = kmeans.labels_[id_path]
#    np.save("./label_data{}/{}.npy".format(label, id_path), data)  
#    

#get_file_number = []
#for root,dirs, files in os.walk('/Users/sisili/Desktop/test_label1'):
#    temp = os.path.basename(files).split('.')[0]
#    get_file_number.append(temp)
  


        
        
