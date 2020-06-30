"""
main pre-processing procedure:
using original cv2.grabCut function, the HED boundaries in SDI paper still nedd to do later
  --TODO grabcut+ and resize the saved image
using region proposal method to generate label for training from input bounding box and image
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for custom database, grabCut algorithm can break when the bounding box's area is too small
and for the big object, like bounding box is just the whole image, it also need fix algorithm
for normal onject, if thE IOU of the region which generated by grabcut and the bounding box < threshold, 
we still need to use the original boudning box as segments 

besides, the coordinate[xmin,ymin,xmax,ymax] may be float decimal, and some of them may be beyond image.shape 
it need fixed to correct int number.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
------------------------------------------------------------------------------------------------------------------
as for the TBbacillus medicinal database,  the outcome seems  poorly... maybe I should only set the pixel-0 to 0.
++++++++++++++++++++++++++++++++++++
+  0 GC_BGD          --background      +
+ 1 GC_FGD           -- foreground     +
+ 2 GC_PR_BGD   --probable background  +
+ 3 GC_PR_FGD   --probable foreground  +
++++++++++++++++++++++++++++++++++++
Based on:
 https://docs.opencv.org/trunk/d8/d83/tutorial_py_grabcut.html

This code is used for grabcut segments of images from train_pairs file,
however, due to lack of segmentation groundtruth, we also grabcut
all images im the dataset.--see grabcut_dataset.py
Same operation also did in box_i.

TODO How about the ICCV2019 new paper:EGNet? Maybe it will help.
  link: https://arxiv.org/pdf/1908.08297v1.pdf
"""

import numpy as np
import cv2 as cv
import os
import sys
import scipy.misc
import matplotlib as mpl
# matplotlib backend
# reference: https://blog.csdn.net/u010945683/article/details/82318832
mpl.use('Agg')  # for 'png' foramt
import matplotlib.pyplot as plt

# multiprocessing grabcut algorithm
# it will cost much time if grabcut image one by one
# reference: https://www.cnblogs.com/taolusi/p/9279675.html
from multiprocessing import Pool 

sys.path.append('util/')
import tbvoc_info
from draw_bbox import get_int_coor

""" file configuration"""
dataset_dir = '/home/JFHEALTHCARE/zhentao.yu/ISBI/TB_SDI_torch/dataset/TBVOC/VOC2019/'
anns_dir = os.path.join(dataset_dir + 'Annotations/')
#you may also need generate val grabcut segments
#if you want to add val loss in train.py
train_pair_dir = os.path.join(dataset_dir, 'train_pairs.txt')
img_dir = os.path.join(dataset_dir, 'JPEGImages/')

# Result of grabcut for each bounding box will be stored at 'Grabcut_inst'
grabcut_dir = os.path.join(dataset_dir, 'Grabcut_inst/')
# Result of grabcut for each image will be stored at 'Segmentation_label'
# because my own dataset has no segmentation groundtruth, so I grabcut all images,
# but the 'Grabcut_inst/' and 'Grabcut_pairs' just has images for training
# I also genererate bbox segments for all images, it will helpful for mIoU calculation
segmentation_label_dir = os.path.join(dataset_dir, 'Segmentation_label_grabcut/')
# Result of grabcut for each image combing with image and bounding box will be stored at 'Grabcut_pairs'
img_grabcuts_dir = os.path.join(dataset_dir, 'Grabcut_pairs/')
# Note: If the instance hasn't existed at 'Grabcut_inst', grabcut.py will grabcut that image

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
               """get int box coor"""
               img_w = img.shape[1]
               img_h = img.shape[0]
               xmin, ymin, xmax, ymax = get_int_coor(xmin, ymin, xmax, ymax, img_w, img_h)           
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
              scipy.misc.toimage(masks[j][0], cmin=0, cmax=255, pal=tbvoc_info.colors_map,
                                                      mode='P' ).save((grabcut_dir).rstrip()+masks[j][1])
        
        """merge masks"""
        # built array(img.shape size)
        mask_ = np.zeros(img.shape[:2])
        for mask in masks:
                mask_ = mask_ + mask[0]
        # save segmetation_label(every object in a image)
        scipy.misc.toimage(mask_, cmin=0, cmax=255, pal=tbvoc_info.colors_map,
                                                mode='P').save((segmentation_label_dir+img_name).rstrip()+'.png')
        
        """create figure with masks and bbox in a image"""
        fig = plt.figure()

        # covert to inch
        # dpi: dot per inch
        W = img.shape[1] / float(fig.get_dpi())
        H = img.shape[0] / float(fig.get_dpi())
        # set fig size
        fig.set_size_inches(W, H)

        for mask in masks:
                rect = mask[2]
                mask = mask[0]
                color = tbvoc_info.colors[np.amax(mask)]
                # add one dimension mask[H,W,1]
                mask = mask[:, :, np.newaxis]
                # draw mask in image(RGB)
                for c in range(3):
                       img[:, :, c] = np.where((mask[:, :, 0] != 0),
                                                                      img[:, :, c]*0.2+0.8*color[c], img[:, :, c])
                #compute bbox coordinates
                #reference: https://www.cnblogs.com/xiaopengli/p/8058408.html
                # use axes, so the coordinates is relative
                left = rect[0] / img.shape[1]
                bottom = 1-  (rect[1] + rect[3]) / img.shape[0]
                ax_w = (rect[0] + rect[2]) / img.shape[1] - left
                ax_h = 1 - rect[1] / img.shape[0] - bottom
                # draw bbox
                ax = fig.add_axes([left, bottom, ax_w, ax_h])
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)
                ax.patch.set_fill(False)
                ax.patch.set_linewidth(5)
                ax.patch.set_color('b')
                # add a non-resampled image to the figure
                # add the axes to the figure
                plt.figimage(img)
        # save img_grabcuts(bbox+mask)
        fig.savefig((img_grabcuts_dir+img_name).rstrip()+'.png')
        """
        plt.cla() : clear local activate axes, others keep constant
        plt.clf() :  clear all axex in the figure, but don't shutdown the figure window
        plt.close(): close the figure window
        """
        plt.cla()
        plt.clf()
        plt.close()


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


