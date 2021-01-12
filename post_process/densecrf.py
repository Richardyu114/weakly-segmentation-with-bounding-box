"""
For get the more smooth masks which predicted by the model.
And the output masks can be the new label to supervise training.

The tool is cited from:
         Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
         Philipp Krähenbühl and Vladlen Koltun
         NIPS 2011

Based on:
   https://github.com/lucasb-eyer/pydensecrf

TODO:
  1.CRF for multi-classification
  2.How to choose the better and appropriate value of para: sxy ans srgb

"""

import numpy as np
import pydensecrf.densecrf as dcrf
import sys
import os
import scipy.misc
from PIL import Image
import cv2 as cv

sys.path.append('util/')
from tbvoc_info import colors_map, colors
from draw_bbox import draw_bbox


def run_densecrf(img_dir, img_name, masks_pro):
        height = masks_pro.shape[0]
        width = masks_pro.shape[1]

        # must use cv2.imread()
        # if use PIL.Image.open(), the algorithm will break
        #TODO --need to fix the image problem
        img = cv.imread(os.path.join(img_dir, img_name).rstrip()+'.jpg')
        # img = cv.resize(img, (1632,1216), interpolation = cv.INTER_LINEAR)
        # for drone-data size 1024
        img = cv.resize(img, (1024,1024), interpolation = cv.INTER_LINEAR)

        # expand to [1,H,W]
        masks_pro = np.expand_dims(masks_pro, 0)
        # masks_pro = masks_pro[:, :, np.newaxis]
        # append to array---shape(2,H,W)
        # one depth represents the class 0, the other represents the class 1
        masks_pro = np.append(1-masks_pro, masks_pro, axis=0)
        #[Classes, H, W]
        # U needs to be flat
        U = masks_pro.reshape(2, -1)
        # deepcopy and the order is C-order(from rows to colums)
        U = U.copy(order='C')
        # for binary classification, the value after sigmoid may be very small
        U = np.where((U < 1e-12), 1e-12, U)
        d = dcrf.DenseCRF2D(width, height, 2)

        # make sure the array be c-order which will faster the processing speed
        # reference: https://zhuanlan.zhihu.com/p/59767914
        U = np.ascontiguousarray(U)
        img = np.ascontiguousarray(img)

        d.setUnaryEnergy(-np.log(U))
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)
        Q = d.inference(5)
        # compare each value between two rows by colum
        # and inference each pixel belongs to which class(0 or 1)
        map = np.argmax(Q, axis=0).reshape((height, width))
        proba = np.array(map)

        return proba

"""model test and test after CRF smooth will both perfrom result saving(masks and pairs)"""
# 先保存预测的mask和图片混合的，保存的时候已经是resize过后的，画框的时候也是直接在这个基础上画，所以要读取原来的图像尺寸进行bndbox的resize
def save_pred_result(img, pred, pred_masks_dir, pred_pairs_dir, box_ann_dir, img_name,ori_imW, ori_imH):
        # save pred_masks
        scipy.misc.toimage(pred, cmin=0, cmax=255, pal=colors_map, mode='P').save(
        os.path.join(pred_masks_dir,img_name).rstrip()+ '.png')
        # save pred_pairs
        # use RGBA alpha channel para and Image.blend function to generate pred_pairs image
        img = img.convert('RGBA')
        pred_img = Image.open(os.path.join(pred_masks_dir,img_name).rstrip()+ '.png')
        pred_img = pred_img.convert('RGBA')
        #img_output = img*(1-alpha)+pred_img*alpha 0.3
        pred_pairs = Image.blend(img, pred_img, 0.4)
        # need to get better quality image TODO
        # resize回原来的 尺寸，看得更清楚一些
        pred_pairs = pred_pairs.resize((ori_imW,ori_imH), Image.LANCZOS)
        pred_pairs.save(os.path.join(pred_pairs_dir, img_name).rstrip()+'.png')
        # draw bbox for show the performance of prediction
        draw_bbox(img_name, box_ann_dir, pred_pairs_dir, pred_pairs_dir, is_resize=False)


if __name__ == '__main__':
    pass
