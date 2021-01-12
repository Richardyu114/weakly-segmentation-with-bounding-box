"""grabcut all images in dataset"""

import numpy as np
import cv2 as cv
import os
import sys
import scipy.misc
import matplotlib as mpl
import matplotlib.pyplot as plt

# multiprocessing grabcut algorithm
# it will cost much time if grabcut image one by one
# reference: https://www.cnblogs.com/taolusi/p/9279675.html
from multiprocessing import Pool 

# matplotlib backend
# reference: https://blog.csdn.net/u010945683/article/details/82318832
mpl.use('Agg')  # for 'png' foramt

sys.path.append('util/')
import tbvoc_info

""" file configuration"""
dataset_dir = '/home/test/weakly_seg/weakly_segmentation/TB_SDI_torch/dataset/TBVOC/VOC2019/'
anns_dir = os.path.join(dataset_dir + 'Annotations/')
train_pair_dir = os.path.join(dataset_dir, 'dataset_pairs.txt')
img_dir = os.path.join(dataset_dir, 'JPEGImages/')

segmentation_label_dir = os.path.join(dataset_dir, 'Segmentation_label_grabcut/')

POOL_SIZE = 4 # multiprocessing para
MINI_AREA = 9 #custom cv2.grabcut break threshold
ITER_NUM = 5  # iterative parameter of grabcut
RECT_SHRINK = 3  # custom parameter for big bounding box 
IOU_THRESHOLD = 0.15  # cue C2 custom parameter in SDI paper section3.1

ANNS = {}

def load_ann():
        with open(train_pair_dir, 'r') as tr_p_txt:
                  for ann_info in tr_p_txt:
                         # split the string line, get the list
                         ann_info = ann_info.rstrip().split('###')
                         if ann_info[0]  not in ANNS:
                             ANNS[ann_info[0]] = []
                         ANNS[ann_info[0]].append(ann_info)
#ann_info:[image name, image name_num_ class_id.png, bbox_ymin,
#                    bbox_xmin,bbox_ymax, bbox_xmax, class_name]

def grabcut(img_name):
        masks = [] 
        # one image has many object that need to grabcut
        for i, ann_info in enumerate(ANNS[img_name], start=1):
               img = cv.imread((img_dir +img_name).rstrip()+'.jpg')
               grab_name = ann_info[1]
               xmin = ann_info[3]
               ymin = ann_info[2]
               xmax = ann_info[5]
               ymax = ann_info[4]
               """get int box coor and fixed some special conditions"""
               tmp = xmin.split('.', 1)
               xmin = int(tmp[0])
               tmp = ymin.split('.', 1)
               ymin = int(tmp[0])
               tmp = xmax.split('.', 1)
               xmax = int(tmp[0])
               tmp = ymax.split('.', 1)
               ymax = int(tmp[0])
               if xmin < 0:
                   xmin = 0
               if ymin < 0:
                   ymin = 0
               if xmax > img.shape[1]:
                   xmax = img.shape[1]
               if ymax > img.shape[0]:
                   ymax = img.shape[0]
                
               box_w = xmax - xmin
               box_h = ymax - ymin
               # cv.grabcut's para
               mask = np.zeros(img.shape[:2], np.uint8)
               # rect is the tuple
               rect = (xmin, ymin, box_w, box_h)
               bgdModel = np.zeros((1, 65), np.float64)
               fgdModel = np.zeros((1, 65), np.float64)
               #for small bbox:
               if box_w * box_h < MINI_AREA:
                   img_mask = mask[ymin:ymax, xmin:xmax] = 1
                # for big box that area == img.area(one object bbox is just the whole image)
               elif box_w * box_h == img.shape[1] * img.shape[0]:
                      rect = [RECT_SHRINK, RECT_SHRINK, box_w - RECT_SHRINK * 2, box_h - RECT_SHRINK * 2]
                      cv.grabCut(img, mask, rect, bgdModel,fgdModel, ITER_NUM, cv.GC_INIT_WITH_RECT)
                      # astype('uint8') keep the image pixel in range[0,255]
                      img_mask =  np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
                # for normal bbox:
               else:
                       cv.grabCut(img, mask, rect, bgdModel,fgdModel, ITER_NUM, cv.GC_INIT_WITH_RECT)
                       img_mask = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
                       # if the grabcut output is just background(it happens in my dataset)
                       if np.sum(img_mask) == 0:
                           img_mask = np.where((mask == 0), 0, 1).astype('uint8')
                        # couting IOU
                        # if the grabcut output too small region, it need reset to bbox mask
                       box_mask = np.zeros((img.shape[0], img.shape[1]))
                       box_mask[ymin:ymax, xmin:xmax] = 1
                       sum_area = box_mask + img_mask
                       intersection = np.where((sum_area==2), 1, 0).astype('uint8')
                       union = np.where((sum_area==0), 0, 1).astype('uint8')
                       IOU = np.sum(intersection) / np.sum(union)
                       if IOU <= IOU_THRESHOLD:
                           img_mask = box_mask
                # for draw mask on the image later           
               img = cv.cvtColor(img, cv.COLOR_BGR2RGB) 
               masks.append([img_mask, grab_name, rect])
        
        num_object = i
        """for multi-objects intersection and fix the label """
        masks.sort(key=lambda mask: np.sum(mask[0]), reverse=True)
        for j in range(num_object):
              for k in range(j+1, num_object):
                      masks[j][0] = masks[j][0] - masks[k][0]
              masks[j][0] = np.where((masks[j][0]==1), 1, 0).astype('uint8')
              """get class name  id"""
              grab_name = masks[j][1]
              class_id = grab_name.split('_')[-1]
              class_id = int(class_id.split('.')[0])

              #set the numpy value to class_id
              masks[j][0] = np.where((masks[j][0]==1), class_id, 0).astype('uint8')
              # save grabcut_inst(one object in a image)
              #scipy.misc.toimage(masks[j][0], cmin=0, cmax=255, pal=tbvoc_info.colors_map,
                                                      #mode='P' ).save((grabcut_dir).rstrip()+masks[j][1])
        
        """merge masks"""
        # built array(img.shape size)
        mask_ = np.zeros(img.shape[:2])
        for mask in masks:
                mask_ = mask_ + mask[0]
        # save segmetation_label(every object in a image)
        scipy.misc.toimage(mask_, cmin=0, cmax=255, pal=tbvoc_info.colors_map,
                                                mode='P').save((segmentation_label_dir+img_name).rstrip()+'.png')
        

def run_grabcut():
        # generate pool for multiprocessing
        p = Pool(POOL_SIZE)
        cnt = 1
        for _ in p.imap_unordered(grabcut, ANNS):
              print('done %d/%d\r' % (cnt, len(ANNS)))
              cnt += 1
        p.close()
        p.join()
       

if __name__ == '__main__':
        load_ann()
        run_grabcut()


