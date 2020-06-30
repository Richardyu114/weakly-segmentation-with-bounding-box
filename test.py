"""
load the trained model to predict images in test_file
generate the predicted masks and use the densecrf to get more smooth masks
besides, calculate the mIoU
-------------------------------------------------------------------------------------
TODO:
   1.yield heatmap
   2.calculate each class's objects number
   3.add bbox on pred_pairs

"""
from datetime import datetime
import os
import sys
import numpy as np
import torch
from torchvision import transforms
import scipy.misc
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('post_process/')
sys.path.append('util/')
sys.path.append('model/')
from densecrf import run_densecrf, save_pred_result
from tbvoc_info import colors_map, colors
# for load model 
from fcn import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
from unet import UNet
import metric

"""file configurations"""
dataset_dir = '/home/JFHEALTHCARE/zhentao.yu/ISBI/TB_SDI_torch/dataset/TBVOC/VOC2019/'
img_dir = os.path.join(dataset_dir, 'JPEGImages/')
test_dir = os.path.join(dataset_dir, 'ImageSets/Main/test.txt')
test_masks_dir = os.path.join(dataset_dir, 'Test_masks/')
test_pairs_dir = os.path.join(dataset_dir, 'Test_pairs/')
crf_masks_dir = os.path.join(dataset_dir, 'CRF_masks/')
crf_pairs_dir = os.path.join(dataset_dir, 'CRF_pairs/')
box_ann_dir = os.path.join(dataset_dir, 'Annotations/')


# define the same image transformations
transformations = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                      mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])])

def test_model():
    """model test configuration"""
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model_fcn.pth')
    #model = torch.load('model_unet.pth')
    model = model.to(device)
    model.eval()
    print(model)

    """testing"""
    img_num = 0
    with open(test_dir, 'r') as test_txt:
         for img_name in test_txt:
             img_num += 1
             # in RGB [W, H, depth]
             img = Image.open(os.path.join(img_dir, img_name).rstrip()+'.jpg')
             img = img.resize((1632, 1216), Image.LANCZOS)
             input_ = transformations(img).float()
             # add batch_size dimension
             #[3, H, W]-->[1, 3, H, W]
             input_ = input_.unsqueeze_(0)
             input_ = input_.to(device)
             pred = model(input_).view([1216, 1632]).data.cpu()
             #pred.shape[H,W]
             pred = np.array(pred)
             test_pred = np.where((pred < 0.5), 0, 1).astype('uint8')
             #save test results
             save_pred_result(img, test_pred, test_masks_dir,
                             test_pairs_dir, box_ann_dir, img_name)
             """crf smooth prediction"""
             crf_pred = run_densecrf(img_dir, img_name,  pred)
             # save crf results
             save_pred_result(img, crf_pred, crf_masks_dir, 
                              crf_pairs_dir, box_ann_dir, img_name)
                     
             print('{} {}.png has been tested and saved the result!'.format
                             (img_num, img_name.rstrip()))





def miou_evaluation():
    mIoU_Bbox, IoU_Bbox, mIoU_Grabcut, IoU_Grabcut = metric.mIoU_without_GT(is_val=False)
    print('(GT is Bbox) mIoU: {}, IoU: {}'.format
                             ('%.3f' %mIoU_Bbox, np.around(IoU_Bbox, decimals=3)))
    print('(GT is Grabcut) mIoU: {}, IoU: {}'.format
                             ('%.3f' %mIoU_Grabcut,  np.around(IoU_Grabcut, decimals=3)))

def acc_evaluation():
    print('calculating the acc of test predictions...')
    average_acc, whole_acc = metric.count_acc(is_crf=False)
    print('average acc is: {}, whole acc is: {}'.format(
         '%.3f' %average_acc, '%.3f' %whole_acc))
    
    print('calculating the acc after crf predictions...')
    average_acc, whole_acc = metric.count_acc(is_crf=True)
    print('average acc is: {}, whole acc is: {}'.format(
         '%.3f' %average_acc, '%.3f' %whole_acc))


if __name__ == '__main__':
     
    print('testing images...') 
    test_model()
    
    acc_evaluation()


    # print('counting the IoU...')
    # miou_evaluation()
                         





                         