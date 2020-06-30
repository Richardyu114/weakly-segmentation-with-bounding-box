"""
from my own dataset to load img and label and send them to model for training

Based on:
 https://github.com/bat67/pytorch-FCN-easiest-demo

TODO:
  1.fix the dimension problem---one hot

"""

from PIL import Image
import numpy as np
import sys
import os
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

"""file configuration"""
dataset_dir = '/home/JFHEALTHCARE/zhentao.yu/ISBI/TB_SDI_torch/dataset/TBVOC/VOC2019/'
train_dir = os.path.join(dataset_dir, 'ImageSets/Main/train.txt')
val_dir = os.path.join(dataset_dir, 'ImageSets/Main/val.txt')
# may use trainval.txt directly, because calculate mIoU of val may useless 
# besides, it can give more instances for model to train
trainval_dir = os.path.join(dataset_dir, 'ImageSets/Main/trainval.txt')
img_dir = os.path.join(dataset_dir, 'JPEGImages/')
# I don't have segmentation groundtruth, so I choose grabcut segments
label_dir = os.path.join(dataset_dir, 'Updated_masks/')


# define image transformations
transformations = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])
                                      ])

"""
get img, label from my own dataset
img: [H,W, 3]
label: [H,W] every value of each pixel is its class_id
Just need to Class Dataset in torch.utils.data and rewrite functions:
 __init__(), __getitem__() and __len__() 
---------------------------------------------------------------------------------------------
!Attention:
You had better use PIL.Image.open() function instead cv2.imread() or io to load the image
The transforms function in pytorch is for PIL format only

"""
class TBDataset(Dataset):
      def __init__(self, txt_dir, width, height,  transform=None):
          self.img_names = []
          with open(txt_dir, 'r') as f_txt:
               for img_name in f_txt:
                   self.img_names.append(img_name)
          
          self.transform = transform
          self.txt_dir = txt_dir
          self.width = width
          self.height = height
                   
      def __getitem__(self, index):
          img_name = self.img_names[index]
          img = Image.open(os.path.join(img_dir, img_name).rstrip()+'.jpg')
          # the resize function like bilinear
          img = img.resize((self.width, self.height), Image.LANCZOS)
          img = np.array(img)
          label = Image.open(os.path.join(label_dir, img_name).rstrip()+'.png')
          # for consider class_id is not consecutive and just fixed by user
          label = label.resize((self.width, self.height), Image.NEAREST)
          label = np.array(label)
          if self.transform is not None:
             img = self.transform(img)
          #img = torch.FloatTensor(img)
          label = torch.FloatTensor(label)
          return img, label
                             


      def __len__(self):
          return len(self.img_names)


# need resize  to upsample, or the dimension is incompatible
# my images' height=1224, 1224 % 32 != 0
train_dataset = TBDataset( train_dir, 1632, 1216, transformations)
val_dataset = TBDataset(val_dir, 1632, 1216, transformations)
trainloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=2,
                                          shuffle=True, num_workers=4)
valloader = torch.utils.data.DataLoader(dataset =val_dataset, batch_size=2, 
                                        shuffle=True, num_workers=4)



if __name__ == '__main__':
    for train_batch in trainloader:
           print(train_batch)
    
    for val_batch in valloader:
           print(val_batch)



