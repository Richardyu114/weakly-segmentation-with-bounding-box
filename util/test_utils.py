from PIL import Image
import numpy
import sys
import os

##configurations
DATA_SRC = "/home/test/weakly_segmentation/TB_SDI_torch/dataset/TBVOC/VOC2019/"
TEST_TEXT = DATA_SRC + "ImageSets/Main/test.txt"
IMAGES_DIR = DATA_SRC + "JPEGImages/"
IMAGES_VISBOX_DIR = DATA_SRC + "visualization_bbox/"
TEST_IMAGES_DIR = DATA_SRC + "test_images/"
TEST_IMAGES_VISBOX_DIR = DATA_SRC + "test_images_with_box/"
TEST_SEG_GT_DIR = DATA_SRC + "test_masks_gt/"

def get_test_imgs_from_txt(test_txt,src_img_dir,img_visbox_dir,dst_test_dir,test_visbox_dir):
    with open(test_txt, 'r') as f_txt:
         img_names = f_txt.readlines()
         num = 0
         total = len(img_names)
         for img_name in img_names:
             num += 1
             img_name = img_name.rstrip() + ".jpg"
             img1 = Image.open(os.path.join(src_img_dir, img_name))
             img2 = Image.open(os.path.join(img_visbox_dir, img_name))
             img1.save(os.path.join(dst_test_dir,img_name))
             img2.save(os.path.join(test_visbox_dir,img_name))
             print("save test images:", num, '/', total, end="\r")


if __name__ == "__main__":
   
   get_test_imgs_from_txt(TEST_TEXT, IMAGES_DIR, IMAGES_VISBOX_DIR,
                          TEST_IMAGES_DIR, TEST_IMAGES_VISBOX_DIR)
   