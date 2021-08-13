# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 20:01:53 2021

@author: smith
"""
import torch
import torch.nn as nn
from torchvision import *
import torch.nn.functional as F
from torchsummary import summary

class architecture():
    def resnet50(out_labels):
        net = models.resnet50(pretrained=True)
        print("model downloaded")
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, out_labels)
        print("Changes DONE")
        print(net)
        return net
    
    def resnet18(out_labels):
        net = models.resnet18(pretrained=True)
        print("model downloaded")
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, out_labels)
        print("Changes DONE")
        print(net)
        return net    
    
    def load_past_model(model,path):
        try:
            model.load_state_dict(torch.load(path))
            print("Past model loaded")
            return model
        except FileNotFoundError:
            print('Invalid file path---> if you are training this model for the first time skip this else check the save path with extension as resnet.pt ---Regards Barbose')
        
    def summary(model,RGB,H,W):       
            summary(model,(R,H,W))
            

class custom_architecture():
    

    '''Defining model'''
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    
    class ImageClassificationBase(nn.Module):
        def training_step(self, batch):
            images, labels = batch 
            out = self(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            return loss
        
        def validation_step(self, batch):
            images, labels = batch 
            out = self(images)                    # Generate predictions
            loss = F.cross_entropy(out, labels)   # Calculate loss
            acc = custom_architecture.accuracy(out, labels)           # Calculate accuracy
            
            return {'val_loss': loss.detach(), 'val_acc': acc}
            
        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
            print('Validation accuracy: %d %%' % (100*epoch_acc.item()))
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        
        def epoch_end(self, epoch, result):
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['val_acc']))
    
    
    class Fruits360CnnModel(ImageClassificationBase):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # output: 32 x 56 x 56
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(5, 5),
# =============================================================================
#     
#                 nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2, 2), # 128x28x28
#                 nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
#                 nn.ReLU(),
#                 nn.MaxPool2d(5, 5), # output: 128 x 5 x 5
#                 
# # =============================================================================
# =============================================================================
#     
#                 nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#                 nn.ReLU(),
#                 nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#                 nn.ReLU(),#output :256*25*25
#                 nn.MaxPool2d(5, 5), # output: 256 x 5 x 5
# =============================================================================
    
                nn.Flatten(), 
                nn.Linear(64*6*6, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 26))
                
            
        def forward(self, xb):
            return self.network(xb)
#%%
