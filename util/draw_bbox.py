# encoding=UTF-8
"""
Draw bounding box on the image
1.visiualize the annotations when see the original dataset
2.draw the bounding box when saving the predicted masks

Based on:
 https://blog.csdn.net/qq_40806289/article/details/90905093
 """

from xml.dom.minidom import parse
import xml.dom.minidom
import os
from PIL import Image, ImageDraw

"""origninal dataset visualization file config"""
# root='/home/zj/yzt/Sputum_Smear.pytorch_3/weakly_segmentation/dataset/tuberculosis-phonecamera/visualizationAnn/'
# annroot='/home/zj/yzt/Sputum_Smear.pytorch_3/weakly_segmentation/dataset/tuberculosis-phonecamera/annotations/abnormal/'
# picroot='/home/zj/yzt/Sputum_Smear.pytorch_3/weakly_segmentation/dataset/tuberculosis-phonecamera/images/'
# dataset_dir = '/home/zj/yzt/Sputum_Smear.pytorch_3/weakly_segmentation/TB_SDI_torch/dataset/TBVOC/VOC2019/ImageSets/Main/dataset.txt'
#label = "TBbacillus"

# bounding box's line color
colormap=['red' , 'green', 'blue' , 'yellow', 'darkorange',  'olive' , 'deeppink']

def get_int_coor(xmin, ymin, xmax, ymax, img_w, img_h):
        xtmp1 = xmin.split('.', 1)
        xmin1 = xtmp1[0]
        xtmp2 = xmax.split('.', 1)
        xmax1 = xtmp2[0]
        xtmp3 = ymin.split('.', 1)
        ymin1 = xtmp3[0]
        xtmp4 = ymax.split('.', 1)
        ymax1 = xtmp4[0]
        xmin = int(xmin1)
        ymin = int(ymin1)
        xmax = int(xmax1)
        ymax = int(ymax1)
        if xmin < 0:
            xmin = 0
        if ymin < 0:
            ymin = 0
        if xmax > img_w:
            xmax = img_w
        if ymax > img_h:
            ymax = img_h
        
        return xmin, ymin, xmax, ymax

"""load annotation info and draw bounding box on image"""
def draw_bbox(img_name, ann_dir,  img_src, img_dst, is_resize=False):
       #using minidom parser to open xml file
       annpath = (ann_dir + img_name).rstrip() + '.xml'
       DOMTree = parse(annpath)
       collection = DOMTree.documentElement
       objects = collection.getElementsByTagName('object')
       # we draw bbox on test_pairs image whose format is PNG
       picpath = (img_src + img_name).rstrip() + '.png'
       # if you draw box in original dataset, the source image format is JPEG
       # picpath = (img_src + img_name).rstrip() + '.jpg'
       img =   Image.open(picpath)
       img_size = collection.getElementsByTagName('size')[0]
       img_w = int(img_size.getElementsByTagName('width')[0].childNodes[0].nodeValue)
       img_h = int(img_size.getElementsByTagName('height')[0].childNodes[0].nodeValue)
       draw = ImageDraw.Draw(img)
       if len(objects) > 0:
           for object_ in objects:
               if object_.getElementsByTagName('name')[0].childNodes[0].nodeValue == 'drone':
                  bndbox = object_.getElementsByTagName('bndbox')[0]
                  xmin = bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue
                  ymin = bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue
                  xmax = bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue
                  ymax = bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue
                  xmin, ymin, xmax, ymax = get_int_coor(xmin, ymin, xmax, ymax, img_w, img_h)
                  # we resized the images [1632, 1224]->[1632, 1216]
                  # so the coordinates also need to change
                  # in drone_data, all images are resized to [1024,1024]
                  if is_resize:
                    #   xmin = int(xmin * 1632 / img_w)
                    #   xmax = int(xmax * 1632 / img_w)
                    #   ymin = int(ymin * 1216 / img_h)
                    #   ymax = int(ymax * 1216 / img_h)
                      xmin = int(xmin * 1024 / img_w)
                      xmax = int(xmax * 1024 / img_w)
                      ymin = int(ymin * 1024 / img_h)
                      ymax = int(ymax * 1024 / img_h)

                  draw.rectangle((xmin, ymin, xmax, ymax), outline = colormap[1],width=2)
           img.save((img_dst + img_name).rstrip() + '.png')


def mkdir(path): 
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)
 

# if __name__ == '__main__':

#     number = 0
#     mkdir(root)
#     with open(dataset_dir, 'r') as f_dataset:
#               for img_name in f_dataset:
#                      draw_bbox(img_name, annroot, picroot, root)
#                      number += 1
#                      print(number, img_name.rstrip())
 

 
 
