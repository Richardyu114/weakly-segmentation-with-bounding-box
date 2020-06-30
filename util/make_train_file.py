"""
when training model, we need the annotations of images from train.txt
the results are saved in train_pairs.txt

Format: 
{image name}###{image name + num + class + .png}###{bbox ymin}
###{bbox xmin}###{bbox ymax}###{bbox xmax}###{class}

Example: 
2011_003038###2011_003038_3_15.png###115
###1###233###136###person(in VOC2012 dataset)

tuberculosis-phone-0001###tuberculosis-phone-0001_3_1.png###397
###1002###494###1006###TBbacillus(in my own dataset)
"""
import os
import numpy as np
from bs4 import BeautifulSoup # using BeautifulSoup parser to get ann_info
from tbvoc_info import tbvoc_classes


dataset_dir = '/home/zj/yzt/Sputum_Smear.pytorch_3/weakly_segmentation/TB_SDI_torch/dataset/TBVOC/VOC2019/'
ann_dir = os.path.join(dataset_dir, 'Annotations/')
train_dir = os.path.join(dataset_dir, 'ImageSets/Main/train.txt')
train_pairs_dir = os.path.join(dataset_dir, 'train_pairs.txt')

def load_class_Bbox(xml_name):
        with open(os.path.join(ann_dir, xml_name), 'r') as ann:
                  soup = BeautifulSoup(ann, 'xml')  

                  #get object classs
                  names = soup.find_all('name')
                  # get Bbox coordinate
                  xmins = soup.find_all('xmin')
                  ymins = soup.find_all('ymin')
                  xmaxs = soup.find_all('xmax')
                  ymaxs = soup.find_all('ymax')

                  ann_info = np.array([])
                  for name, xmin, ymin, xmax, ymax in zip(names, xmins, 
                                                                                         ymins, xmaxs, ymaxs):
                         ann_info = np.append(ann_info, np.array([name.string, xmin.string, 
                                                                        ymin.string, xmax.string, ymax.string]))
                  ann_info = ann_info.reshape(-1, 5)

        return ann_info

def train_pairs_file():
        num = 0
        with open(train_pairs_dir, 'w+') as tp, open(train_dir, 'r') as t:
                   for img_name in t:
                          num += 1
                          img_name = img_name.rstrip()   #delete the space
                          ann_info = load_class_Bbox(img_name + '.xml')
                          for i, info in enumerate(ann_info):
                                 if info[0] in tbvoc_classes:
                                     obj_name = '{}_{}_{}.png'.format(img_name,
                                                          i, tbvoc_classes[info[0]])
                                     tp.write('{}###{}###{}###{}###{}###{}###{}\n'.format(img_name, 
                                                        obj_name, info[2], info[1], info[4], info[3], info[0] ))
                          print(num, img_name)  # show the saving process


if __name__ == "__main__":
    train_pairs_file()
                          
                   
                   
