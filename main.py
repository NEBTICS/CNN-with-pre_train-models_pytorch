# -*- coding: utf-8 -*-
"""
@author: smith
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import  DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as tt
import torch.nn.functional as F

import os

data_dir = r'C:\Users\smith\Videos\UCF101\AutoEncoder-Video-Classification-master'
#%%
print(os.listdir(data_dir)) #folders in the dataset folder
classes = os.listdir(data_dir + "/test")
classes_list=classes
print(classes)
#%%
'''Lets see how many images are there in per category '''
#create a dictionary to hold the count
train_image_count = {}
for item in classes:
    train_image_count[item] = 0
train_image_count
#now let us find the count
for item in classes:
    train_image_count[item] = len( os.listdir(data_dir + "/test/"+item))
#printing
print(train_image_count)
   
 #%%   
plt.bar(list(train_image_count.keys()), train_image_count.values(), color='g')
plt.show()

#%%
'''Data transforms '''
mean = [0.4690646, 0.4407227, 0.40508908]
std = [0.2514227, 0.24312855, 0.24266963]


train_trams=tt.Compose([tt.Resize((224,224)),
                        tt.RandomHorizontalFlip(),tt.ToTensor(),tt.Normalize(mean,std,inplace=True)])
#test
test_trams=tt.Compose([tt.Resize((224,224)),tt.ToTensor(),tt.Normalize((0.5),(0.5),inplace=True)])


#%%
''' datasets loading'''
train_ds=ImageFolder(data_dir+'/train',train_trams)
val_ds=ImageFolder(data_dir+'/test',test_trams)


#%%
batch_size=96 #for 32 #2.1 gb 64 3.3GB 
train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size, num_workers=0, pin_memory=True)
print(len(train_dl),len(val_dl))    

#%%
import GPU 
from GPU import gpu
device = gpu.get_default_device()
print(device)
#%%


#%%
import architecture
from architecture import architecture
model=architecture.resnet18(19)
#%%


#%%
path='A:/BE/HAR/resnet.pt'
mdoel=architecture.load_past_model(model, path)

#%%


train_dl=gpu.DeviceDataLoader(train_dl, device)
val_dl=gpu.DeviceDataLoader(val_dl, device)
gpu.to_device(model, device)


#%%

n_epochs = 1
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


#%%
'''Traning'''
import train
from train import resnet_train
#%%
resnet_train.train(train_dl,val_dl,model,device,optimizer,criterion,n_epochs)


#%%
from CONFUSION_MATRIX import confusion_matrix

confusion_matrix(classes_list, val_dl, model, device)

