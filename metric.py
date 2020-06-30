"""
####use IoU and acc metric only for now###

calculate the mIoU of the result that model output on val.txt and test.txt.
formula: IoU = intersection / union
'intersection' is the number of pixel that model predict correctly
'union' is the sum of the number pixel that model predict wrong 
  and what the pixel should be(right + wrong) at the same time
----------------------------------------------------------------------------------------------- 
if the dataset has not segmentation groundtruth(for most weakly bbox supervisied sgmentation),
use the bbox and the grabcut segments to calculate the IoU respectively; 
---TODO using FROC metric
s
"""

import os
import sys
import numpy as np
from PIL import Image

from xml.dom.minidom import parse
import xml.dom.minidom
sys.path.append('util/')
from draw_bbox import get_int_coor
# import tbvoc_info


"""file configuration"""
dataset_dir = '/home/JFHEALTHCARE/zhentao.yu/ISBI/TB_SDI_torch/dataset/TBVOC/VOC2019/'
val_dir = os.path.join(dataset_dir, 'ImageSets/Main/val.txt')
test_dir = os.path.join(dataset_dir, 'ImageSets/Main/test.txt')
val_pred_dir = os.path.join(dataset_dir, 'Val_masks/')
test_pred_dir = os.path.join(dataset_dir, 'Test_masks/')
crf_pred_dir = os.path.join(dataset_dir, 'CRF_masks/')
img_dir = os.path.join(dataset_dir, 'JPEGImages/')
anns_dir = dataset_dir + 'Annotations/'
# segmentation groundtruth dir(for all images)
GT_seg_dir = os.path.join(dataset_dir, 'SegmentationClass/')
# bbox segments dir (for all images)
Bbox_seg_dir = os.path.join(dataset_dir, 'Segmentation_label_box/')
# grabcut  segments dir (for all images)
Grabcut_seg_dir = os.path.join(dataset_dir, 'Segmentation_label_grabcut/')
# number of classes
NUM_CLASSES = 2
# ignore regions label id
IGNORE_REG_ID = 2



def count_mIoU(txt_dir, pred_dir, seg_dir):
        Intersection = np.zeros(NUM_CLASSES, dtype=int)
        Union = np.zeros(NUM_CLASSES, dtype=int)    
        with open(txt_dir, 'r') as f_txt:
                  for img_name in f_txt:
                         img_name = img_name.rstrip()
                         #img.size [W,H]
                         img = Image.open(seg_dir+img_name+'.png')
                         # resize the groundtruth label
                         # because the pred mask is different size 
                         img = img.resize((1632, 1216), Image.NEAREST)
                         #GT_seg.shape [H,W]
                         # every value in array is class_id
                         seg = np.array(img)
                         #  change the ignore region label to background label
                         seg = np.where((seg == IGNORE_REG_ID), 0, seg)
                         img_height, img_width = seg.shape
                         pred_img = Image.open(pred_dir+img_name+'.png')
                         pred_seg = np.array(pred_img)
                         # change the ignore region label to background
                         pred_seg = np.where((pred_seg == IGNORE_REG_ID), 0, pred_seg)
                         for i in range(img_height):
                                for j in range(img_width):
                                       if pred_seg[i][j] == seg[i][j]:
                                           # add one on the value of class_id index 
                                           Intersection[pred_seg[i][j]] += 1
                                           Union[pred_seg[i][j]] += 1
                                       else:
                                            Union[pred_seg[i][j]] += 1
                                            Union[seg[i][j]] += 1
                         print(img_name)

        IoU = np.divide(Intersection, Union, out=np.ones(NUM_CLASSES), where=Union!=0)

        mIoU = np.mean(IoU)

        return mIoU, IoU
       
       
def mIoU_with_GT(is_val=False):
        if is_val:
           txt_dir = val_dir
           pred_dir = val_pred_dir
        else:
           txt_dir = test_dir
           pred_dir = crf_pred_dir

        mIoU_GT, IoU_GT = count_mIoU(txt_dir, pred_dir, GT_seg_dir)

        return mIoU_GT, IoU_GT
       
        

def mIoU_without_GT(is_val=False):
        if is_val:
           txt_dir = val_dir
           pred_dir = val_pred_dir
        else:
           txt_dir = test_dir
           pred_dir = crf_pred_dir

        mIoU_Bbox, IoU_Bbox = count_mIoU(txt_dir, pred_dir, Bbox_seg_dir)
        mIoU_Grabcut, IoU_Grabcut = count_mIoU(txt_dir, pred_dir, Grabcut_seg_dir)

        return mIoU_Bbox, IoU_Bbox, mIoU_Grabcut, IoU_Grabcut

      
"""
we use the acc to evaluate the predictions with box annotations only
if the predicted mask(connected component) hit the box, we think it's right
acc = right num / box num
this metric only be used for tested image:
averge_acc + whole_acc
"""
def count_acc(is_crf=True):
    average_acc = 0
    whole_acc = 0
    #box num of per image
    box_num_img = 0
    #box num of whole  test images
    box_num_imgs = 0
    #right predictions num per image
    right_num_img = 0
    #right predictions num of whole test images
    right_num_imgs = 0
    #num of test images
    num_imgs = 0
    if is_crf:
       pred_dir = crf_pred_dir
    else:
       pred_dir = test_pred_dir
    with open (test_dir, 'r') as f_test_txt:
         for img_name in f_test_txt:
             img_name = img_name.rstrip()
             num_imgs += 1
             img = Image.open(img_dir+img_name+'.jpg')
             img_w = img.size[0]
             img_h = img.size[1]
             pred_mask = Image.open(pred_dir+img_name+'.png')
             pred_mask = np.array(pred_mask)

             annpath = (anns_dir + img_name).rstrip() + '.xml'
             DOMTree = parse(annpath)
             collection = DOMTree.documentElement
             objects = collection.getElementsByTagName('object')
             if len(objects) > 0:
                for object_ in objects:
                  bndbox = object_.getElementsByTagName('bndbox')[0]
                  xmin = bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue
                  ymin = bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue
                  xmax = bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue
                  ymax = bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue
                  xmin, ymin, xmax, ymax = get_int_coor(xmin, ymin, xmax, ymax, img_w, img_h)
                  xmin = int(xmin * 1632 / img_w)
                  xmax = int(xmax * 1632 / img_w)
                  ymin = int(ymin * 1216 / img_h)
                  ymax = int(ymax * 1216 / img_h)
                  box_num_img += 1
                  box_num_imgs += 1
                  if np.sum(pred_mask[ymin:ymax, xmin:xmax]) != 0:
                     right_num_img += 1
             average_acc += (right_num_img / box_num_img)
             right_num_imgs += right_num_img
             right_num_img = 0
             box_num_img = 0

             print('{} / {}'.format(num_imgs, img_name), end='\r')

         average_acc = average_acc / num_imgs
         whole_acc = right_num_imgs / box_num_imgs

    return average_acc, whole_acc
             



if __name__ == '__main__':

    mIoU_Bbox, IoU_Bbox, mIoU_Grabcut, IoU_Grabcut = mIoU_without_GT(is_val=False)
    print(mIoU_Bbox)
    #    for x, y in zip(tbvoc_info.tbvoc_classes.keys(), IoU):
    #           print(x, y)








