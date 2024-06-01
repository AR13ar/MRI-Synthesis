
import glob
import numpy as np
import random
import fnmatch
import os
from PIL import Image
from matplotlib import pyplot as plt
import PIL
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d,MaxUnpool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from skimage import util
from torchvision import transforms, models
from torch import optim


path = "\\MRI_synthesis\\train"
list_data = os.listdir(path)

class data_set(Dataset):

  def __init__(self, file_paths, transform1):
    self.file_paths =  file_paths
    self.transform1 = transform1

  def __len__(self):
    return len(self.file_paths)

  def __getitem__(self, indx):
    img = Image.open(self.file_paths[indx]).convert('L')
    img1 = self.transform1(img)
    return img1


train_path =[]
for i in range(len(list_data)):
  train_path.append(path + '/' + list_data[i])



training_data_gd = data_set(file_paths = train_path,
                                           transform1 = transforms.Compose([
                                               transforms.CenterCrop((180,180)),
                                               transforms.Resize((100, 100)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0],[1])
                                               ])
                                           )

dataloader_train_gd = torch.utils.data.DataLoader(training_data_gd, batch_size = 32,
                        shuffle=True)

    
class model_all(nn.Module):
  def __init__(self, mod):
    super(model_all, self).__init__()
    self.mod = mod

    self.generator = Sequential(nn.ConvTranspose2d(in_channels=100,
                                                out_channels= 512,
                                                kernel_size= 4,
                                                stride= 2,
                                                padding= 0 ),
                                nn.BatchNorm2d(512),
                                nn.ReLU(True),
                                nn.ConvTranspose2d(in_channels= 512,
                                                out_channels= 256,
                                                kernel_size= 4,
                                                stride= 4,
                                                padding=1),
                                nn.BatchNorm2d(256),
                                nn.ReLU(True),
                                nn.ConvTranspose2d(in_channels= 256,
                                                out_channels= 128,
                                                kernel_size= 4,
                                                stride= 2,
                                                padding = 2),
                                nn.BatchNorm2d(128),
                                nn.ReLU(True),
                                nn.ConvTranspose2d(in_channels= 128,
                                                out_channels= 1,
                                                kernel_size= 4,
                                                stride= 4,
                                                padding = 2),
                                nn.Tanh()
                                )

    self.discriminator = Sequential(nn.Conv2d(in_channels=1,
                                                out_channels= 16,
                                                kernel_size= 4,
                                                stride=2,
                                                padding=1, bias=False),
                                nn.BatchNorm2d(16),
                                nn.LeakyReLU(0.2, inplace=True),
                                nn.Conv2d(in_channels=16,
                                                out_channels= 16,
                                                kernel_size= 4,
                                                stride=2,
                                                padding=1, bias=False),
                                nn.BatchNorm2d(16),
                                nn.LeakyReLU(0.2, inplace=True),
                                
                                nn.Conv2d(in_channels=16,
                                                out_channels= 32,
                                                kernel_size= 4,
                                                stride=2,
                                                padding=1, bias=False),
                                nn.BatchNorm2d(32),
                                nn.LeakyReLU(0.2, inplace=True),
                                
                                nn.Conv2d(in_channels=32,
                                                out_channels= 16,
                                                kernel_size= 2,
                                                stride=2,
                                                padding=1, bias=False),
                                nn.BatchNorm2d(16),
                                nn.LeakyReLU(0.2, inplace=True),
                                

                                 )

    self.classifier = nn.Sequential(
                                nn.Linear(16*7*7, 1, bias=False),
                                # nn.ReLU(inplace=True),
                                # nn.Linear(32, 1, bias=False),
                                # nn.ReLU(inplace=True),
                                # nn.Linear(64, 1, bias=False) ,
                                nn.Sigmoid())

  def forward(self, x):

    if (self.mod == "Gen"):
      x = self.generator(x)
      return x

    elif (self.mod == "Dis"):
      x = self.discriminator(x)
      # print(x.shape)
      x = x.view(-1, 16*7*7)
      x = self.classifier(x)
      return x

model3 = model_all("Gen")
model4 = model_all("Dis")


#sanity check
# fixed_noise = torch.randn(1, 100, 1, 1)
# out = model3(fixed_noise)
# print(out[0][0].detach())
# test_data = torch.randn(1, 1, 100, 100)
# out = model4(test_data)
# print(out.shape)


gen_optimizer = torch.optim.Adam(model3.parameters(), lr = 0.0003)
dis_optimizer = torch.optim.Adam(model4.parameters(), lr = 0.0003)

scheduler_gen = torch.optim.lr_scheduler.StepLR(gen_optimizer, step_size=200, gamma=0.1)
scheduler_dis = torch.optim.lr_scheduler.StepLR(dis_optimizer, step_size=100, gamma=0.1)

criterian_loss = nn.MSELoss()
criterian_binary_classification = nn.BCELoss()

n_epochs = 500

for epoch in range(n_epochs):
  train_loss_gen = 0.0
  train_loss_dis = 0.0
  for data in dataloader_train_gd:
        
      batch_size = data.shape[0]
      label_rp = torch.tensor(np.ones((batch_size,1)))
      ''' 
        Discriminator Training
      '''
      ''' Part 1
      '''
      dis_optimizer.zero_grad()
      true_out = model4(data)
      loss_dis_1 = criterian_loss(true_out, label_rp.float())
      # print(loss_dis_1)
      loss_dis_1.backward()
      
      ''' Part 2
      '''
      fixed_noise = torch.randn(batch_size, 100, 1, 1)
      out = model3(fixed_noise)
      label_rp_zeros = torch.tensor(np.zeros((batch_size,1)))
      dis_gen_out = model4(out.detach())

      loss_dis1 = criterian_loss(dis_gen_out, label_rp_zeros.float())
      # loss_dis2 = criterian_binary_classification(dis_gen_out, label_rp_zeros.float())
      loss_dis_2 =  loss_dis1#loss_dis1 +)/2

      train_loss_dis += loss_dis_2/2 + loss_dis_1/2 
      # print(loss_dis_2)
      loss_dis_2.backward()
      dis_optimizer.step()
      
      

      
      ''' Generator Training
      '''
      gen_optimizer.zero_grad()

      label_dis_p = model4(out)
      
      loss_1 = criterian_loss(label_rp.float(), label_dis_p)
      # loss_2 = criterian_binary_classification(label_rp.float(), label_dis_p)
      # print(label_dis_p)
      loss_gen = loss_1 #(loss_1 +)/2
      train_loss_gen += loss_gen
      # print(loss_gen)
      loss_gen.backward()
      gen_optimizer.step()
      # scheduler_gen.step()
     
      # scheduler_dis.step()
  print('Epoch: {} \tTraining Loss(Gen): {:.4f} \tTraining Loss(Dis): {:.4f}'.format(epoch, train_loss_gen, train_loss_dis))      


path_save = "MRI_synthesis\\generated"

for i in range(50):
    fixed_noise_infer = torch.randn(1, 100, 1, 1)
    out_infer = model3(fixed_noise_infer)
    plt.imshow(out_infer.detach().squeeze(), cmap = 'gray')
    #plt.savefig(path_save  +'\\' + str(i) + "_generated"  + '.png' )


