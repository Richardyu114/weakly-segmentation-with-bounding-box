"""
After certain epoch,  updating the segmentation labels for supervised learning
Use the model and denseCRF to predict train dataset 
  (as for val dataset, we just use it to watch if overfitting or not)
TODO how to update multi-classes object masks?
-------------------------------------------------------------------------------------------
update rules: TODO--need to think clearly
   1.the value of num_epoch for iteration (this para is used in train.py)
   2.if the predicted segment is false-positive, it should throw away
   3.if there are missed diagnosis, we will use the grabcut or box_i segments
   4.if the predicted segment's some area is beyond the Bbox, should it be reset to 0?
   5. update the whole dataset's labels(because the val dataset need too)
-------------------------------------------------------------------------------------------

"""

import os
import sys
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import scipy.misc

sys.path.append('util/')
sys.path.append('model/')
from densecrf  import run_densecrf
from tbvoc_info import colors_map
from draw_bbox import get_int_coor
from fcn import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
from unet import UNet

dataset_dir = '/home/test/weakly_seg/weakly_segmentation/TB_SDI_torch/dataset/TBVOC/VOC2019/'
val_pred_dir = os.path.join(dataset_dir, 'Val_masks/')
# put the masks into label_dir, even include the first grabcut (or box_i) segments
label_dir = os.path.join(dataset_dir, 'Updated_masks/')
dataset_pairs_dir = os.path.join(dataset_dir, 'dataset_pairs.txt')
dataset_txt_dir = os.path.join(dataset_dir, 'ImageSets/Main/dataset.txt')
img_dir = os.path.join(dataset_dir, 'JPEGImages/')


def update_label(predict_model, device):
       
       """load train_pairs.txt info for check the missed diagnosis objects"""
       #ann_info:[image name, image name_num_ class_id.png, bbox_ymin,
       #                    bbox_xmin,bbox_ymax, bbox_xmax, class_name]
       print('start to update...')
       ANNS = {}
       with open(dataset_pairs_dir, 'r') as da_p_txt:
                 for ann_info in da_p_txt:
                        # split the string line, get the list
                        ann_info = ann_info.rstrip().split('###')
                        if ann_info[0].rstrip()  not in ANNS:
                            ANNS[ann_info[0].rstrip()] = []
                        ANNS[ann_info[0].rstrip()].append(ann_info)


    #    print('loading model...')
    #    predict_model = torch.load(model_path)
    #    device = torch.device('cuda: 3' if torch.cuda.is_available() else 'cpu')
    #    predict_model = predict_model.to(device)
       predict_model.eval()

       

       # define the same image transformations
       transformations = transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                             std=[0.229, 0.224, 0.225])
                                             ])

       update_num = 0
       print('updating progress:')
       with open(dataset_txt_dir, 'r') as da_txt:
                 # don't use the code line below
                 # or it will close the file and the whole programm end here (I guess)
                 # I debug here for two hours......
                 #lines = len(da_txt.readlines())
                 for update_name in da_txt:
                         update_num += 1
                         # in RGB [W, H, depth]
                         img = Image.open(os.path.join(img_dir, update_name).rstrip()+'.jpg')
                         img_w = img.size[0]
                         img_h = img.size[1]
                         #img = img.resize((1632, 1216), Image.LANCZOS)
                         # in drone_data,size-1024
                         img = img.resize((1024, 1024), Image.LANCZOS)
                         input_ = transformations(img).float()
                         # add batch_size dimension
                         #[3, H, W]-->[1, 3, H, W]
                         input_ = input_.unsqueeze_(0)
                         input_ = input_.to(device)
                         #pred = predict_model(input_).view([1216, 1632]).data.cpu()
                         # in drone data, size-1024
                         pred = predict_model(input_).view([1024, 1024]).data.cpu()
                         #pred.shape[H,W]
                         pred = np.array(pred)
                         """crf smooth prediction"""
                         crf_pred = run_densecrf(img_dir, update_name,  pred)

                         """start to update"""
                         last_label = Image.open(os.path.join(label_dir, update_name).rstrip()+'.png')
                         #last_label = last_label.resize((1632, 1216), Image.NEAREST)
                         # in drone_data size-1024
                         last_label = last_label.resize((1024, 1024), Image.NEAREST)
                         last_label = np.array(last_label)

                         # predicted label without false-positive segments
                         updated_label = crf_pred + last_label
                         updated_label = np.where((updated_label==2), 1, 0).astype('uint8')
                         # predicted label with missed diagnosis 
                         # we just use the box segments as missed diagnosis for now
                         info4check = ANNS[update_name.rstrip()]
                        #  masks_missed = np.zeros((1216, 1632), np.uint8)
                        # for drone data size-1024
                         masks_missed = np.zeros((1024, 1024), np.uint8)
                         for box4check in info4check:
                                xmin = box4check[3]
                                ymin = box4check[2]
                                xmax = box4check[5]
                                ymax = box4check[4]
                                xmin, ymin, xmax, ymax = get_int_coor(xmin, ymin, 
                                                                      xmax, ymax, img_w, img_h)
                                # xmin = int(xmin * 1632 / img_w)
                                # xmax = int(xmax * 1632 / img_w)
                                # ymin = int(ymin * 1216 / img_h)
                                # ymax = int(ymax * 1216 / img_h)
                                # for drone data - size 1024
                                xmin = int(xmin * 1024 / img_w)
                                xmax = int(xmax * 1024 / img_w)
                                ymin = int(ymin * 1024 / img_h)
                                ymax = int(ymax * 1024 / img_h)
                                if np.sum(updated_label[ymin:ymax, xmin:xmax]) == 0:
                                    masks_missed[ymin:ymax, xmin:xmax] = 1

                         updated_label = updated_label + masks_missed
                         scipy.misc.toimage(updated_label, cmin=0, cmax=255, pal=colors_map, 
                                                            mode='P').save(os.path.join(label_dir, 
                                                                           update_name).rstrip()+ '.png')
                         print('{} / {}'.format(update_num, len(ANNS)), end='\r')



if __name__ == '__main__':
    
    pass
    # update_label('model_fcn.pth')


