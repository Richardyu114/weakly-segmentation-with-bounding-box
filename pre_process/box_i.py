"""
use box and boxi methods to get the segment from the SDI paper
box method: all the pixel of inner box are the object 
if two object overlapping, the small area will be the another objetc class instead the bigger one
boxi method: introduce the ignore region
"instead of using filled rectangles as initial labels, we fill in the 20% inner region, 
and leave the remaining inner area of the bounding box as ignore regions"
For more information, please read the paper.
----------------
but in this code, we just shrink the box and don't use ignore label
"""

import os
import numpy as np
# bs4 reference：https://beautifulsoup.readthedocs.io/zh_CN/v4.4.0/
from bs4 import BeautifulSoup
import scipy.misc
from PIL import Image 
import sys

sys.path.append('util/')
import tbvoc_info

"""output file configuration"""
dataset_dir = '/home/JFHEALTHCARE/zhentao.yu/ISBI/TB_SDI_torch/dataset/TBVOC/VOC2019/'
anns_dir = dataset_dir + 'Annotations/'
segmentation_label_dir = os.path.join(dataset_dir, 'Segmentation_label_boxi/')
# you may alse need to generate val box_i segments
# if you want to add val loss in train.py
train_dir = dataset_dir + 'ImageSets/Main/train.txt'


def get_box_info(ann_file):
    with open((anns_dir + ann_file).rstrip() +'.xml', 'r') as ann:
         soup = BeautifulSoup(ann, 'xml')
         # get image size
         size = soup.find('size')
         width = int(size.find('width').string)
         height = int(size.find('height').string) 
         # [H, W], when saveing, the shape wiil be [W,H]
         mask = np.zeros((height, width), np.uint8) 
         box_anns = []

         objects = soup.find_all(['object']) #get the 'object' tree list
         for object_ in objects:
             # get the class id
             name = object_.find('name').string

             if name not in tbvoc_info.tbvoc_classes:
                continue
             class_id = tbvoc_info.tbvoc_classes[name]

             # get box coor and fix some special conditions
             xmin = object_.find('xmin').string
             ymin = object_.find('ymin').string
             xmax = object_.find('xmax').string
             ymax = object_.find('ymax').string
            #he coordinate[xmin,ymin,xmax,ymax] may be float decimal
             tmp = xmin.split('.', 1)
             xmin = int(tmp[0])
             tmp = ymin.split('.', 1)
             ymin = int(tmp[0])
             tmp = xmax.split('.', 1)
             xmax = int(tmp[0])
             tmp = ymax.split('.', 1)
             ymax = int(tmp[0])
             """box coor may be beyond the image boundary"""
             if xmin < 0:
                xmin = 0
             if ymin < 0:
                ymin = 0
             if xmax > int(width):
                xmax = int(width)
             if ymax > int(height):
                ymax = int(height)

             box_w = xmax - xmin
             box_h = ymax - ymin
             area = box_w * box_h
            
            # the shrink para can change to another value for your own dataset
            # use about 80% orignial box area as segments
             xmin_ign = int(xmin + box_w * 0.05) 
             ymin_ign = int(ymin + box_h * 0.05)
             xmax_ign = int(xmax - box_w * 0.05)
             ymax_ign = int(ymax - box_h * 0.05)

             box_anns.append([area, mask, xmin, ymin, xmax, ymax, xmin_ign,
                         ymin_ign, xmax_ign, ymax_ign, class_id])
         # thinking about the intersection of multi-objects               
         box_anns.sort(reverse=True)

         return box_anns

def box_to_sge():
    num = 0
    with open(train_dir, 'r') as tr_txt:
         for ann_file in tr_txt:
             box_anns = get_box_info(ann_file)
             num += 1
             for box_ann in box_anns:
                 mask = box_ann[1]
                 mask[box_ann[3]:box_ann[5], box_ann[2]:box_ann[4]] = box_ann[-1]
             # be sure the scipy_version < 1.0.0, 
             # it deprecated in SciPy 1.0.0, and will be removed in 1.2.0   
             # use PIL.Image.formarray() instead in later version
             mask = scipy.misc.toimage(mask, cmin=0, cmax=225, 
                                   pal=tbvoc_info.colors_map, mode='P')
             mask.save((segmentation_label_dir+ann_file).rstrip()+'.png')
             print(num, ann_file)

def boxi_to_sge():
    num = 0
    with open(train_dir, 'r') as tr_txt:
         for ann_file in tr_txt:
             box_anns = get_box_info(ann_file)
             num += 1
             for box_ann in box_anns:
                 mask = box_ann[1]
                 # if use ignore regions, give label 2
                 # or use label 0
                 #mask[box_ann[3]:box_ann[5], box_ann[2]:box_ann[4]] = 2
                 mask[box_ann[3]:box_ann[5], box_ann[2]:box_ann[4]] = 0
                 # class label
                 mask[box_ann[7]:box_ann[9], box_ann[6]:box_ann[8]] = box_ann[-1]
             # be sure the scipy_version <= 1.1.0, or this function is removed   
             mask = scipy.misc.toimage(mask, cmin=0, cmax=225, 
                                   pal=tbvoc_info.colors_map, mode='P')
             mask.save((segmentation_label_dir+ann_file).rstrip()+'.png')
             print(num, ann_file)

if __name__ == '__main__':
    #box_to_sge()
    boxi_to_sge()        

         
         

    