"""
Based on:
https://www.cnblogs.com/blog4ljy/p/9195752.html

####################################################
PASCAL VOC2012 annotation.xml format:
annotation
|--folder
|--filename
|--source
     |--database
     |--annotation
     |--image
|--size
     |--width
     |--height
     |--depth
|--segmented
|--object
     |--name
     |--pose
     |--truncated
     |--difficulted
     |--bndbox
         |--xmin
         |--ymin
         |--xmax
         |--ymax
|--object  #if there have many objects
     |--      
     .......
###################################################################
"""

from xml.dom.minidom import parse, parseString
import xml.dom.minidom
import os
from lxml.etree import Element, SubElement, tostring

dataset_dir ='/home/zj/yzt/Sputum_Smear.pytorch_3/weakly_segmentation/TB_SDI_torch/dataset/TBVOC/VOC2019/'
ann_src = '/home/zj/yzt/Sputum_Smear.pytorch_3/weakly_segmentation/dataset/tuberculosis-phonecamera/annotations/abnormal/'
ann_dst =  dataset_dir + 'Annotations/'
anns = os.listdir(ann_src)
file_dst = os.path.exists(ann_dst)
if not file_dst:
    os.makedirs(ann_dst)

"""TB_VOC Annotations files'  some fixed configurations: """
database = 'Makerere Automated Lab Diagnostics Database'
annotation = 'Makerere University'
image = 'Mulago National Referral Hospital'
folder = 'VOC2019'  #custom folder name
label = 'TBbacillus'
width = '1632'
height = '1224'
depth = '3'


num = 0 #see the processing speed

for ann in anns:
    num += 1
    print (num, ann)
    annpath=ann_src + ann

    filename = ann.replace('xml', 'jpg')
    # annotation
    node_root = Element('annotation')
    ## folder
    node_folder = SubElement(node_root, 'folder')
    node_folder.text  = folder
    ## filename
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = filename
    ## source
    node_source = SubElement(node_root, 'source')
    node_database = SubElement(node_source, 'database')
    node_database.text = database
    node_annotation = SubElement(node_source, 'annotation')
    node_annotation.text = annotation
    node_image = SubElement(node_source, 'image')
    node_image.text = image
    ## size
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = width
    node_height = SubElement(node_size, 'height')
    node_height.text = height
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = depth
    ## segmented
    node_segmented = SubElement(node_root, 'segmented')
    node_segmented.text = '1'

    """using minidom parser to open xml file"""
    DOMTree = parse(annpath)
    collection = DOMTree.documentElement
    objects = collection.getElementsByTagName('object')
    for object_ in objects:
         pose = object_.getElementsByTagName('pose')[0].childNodes[0].nodeValue
         truncated = object_.getElementsByTagName('truncated')[0].childNodes[0].nodeValue
         occluded =  object_.getElementsByTagName('occluded')[0].childNodes[0].nodeValue
         
         bndbox = object_.getElementsByTagName('bndbox')[0]
         xmin = bndbox.getElementsByTagName('xmin')[0].childNodes[0].nodeValue
         ymin = bndbox.getElementsByTagName('ymin')[0].childNodes[0].nodeValue
         xmax = bndbox.getElementsByTagName('xmax')[0].childNodes[0].nodeValue
         ymax = bndbox.getElementsByTagName('ymax')[0].childNodes[0].nodeValue
        ## object
         node_object = SubElement(node_root, 'object')
         node_name = SubElement(node_object, 'name')
         node_name.text = label
         node_pose = SubElement(node_object, 'pose')
         node_pose.text = pose
         node_truncated = SubElement(node_object, 'truncated')
         node_truncated.text = truncated
         node_difficult = SubElement(node_object, 'difficult')
         node_difficult.text = occluded
         node_bndbox = SubElement(node_object, 'bndbox')
         node_xmin = SubElement(node_bndbox, 'xmin')
         node_xmin.text  = xmin
         node_ymin = SubElement(node_bndbox, 'ymin')
         node_ymin.text = ymin
         node_xmax = SubElement(node_bndbox, 'xmax')
         node_xmax.text = xmax
         node_ymax = SubElement(node_bndbox, 'ymax')
         node_ymax.text = ymax

    xml = tostring(node_root, pretty_print=True)  #display in appropriate format && line feed
    dom = parseString(xml)
    save_xml = os.path.join(ann_dst, ann)

    with open(save_xml, 'wb') as f:
           f.write(xml)

