import os
import random

"""train-n*0.8, val-n*0.1, test-n*0.1"""
def _main():
    trainval_percent = 0.9
    train_percent = 0.9
    dataset_dir = '/home/zj/yzt/Sputum_Smear.pytorch_3/weakly_segmentation/TB_SDI_torch/dataset/TBVOC/VOC2019/'
    xml_ann_src = os.path.join(dataset_dir, 'Annotations/')
    xml_anns = os.listdir(xml_ann_src)

    num = len(xml_anns)
    list = range(num)
    num_trv = int(num * trainval_percent)  #n*0.9-->trainval
    num_tr = int(num_trv * train_percent)  #n*0.9*0.9-->train
    list_trv = random.sample(list, num_trv)  #get indices
    list_tr = random.sample(list_trv, num_tr)

    f_trainval = open(os.path.join(dataset_dir, 'ImageSets/Main/trainval.txt'), 'w+')
    f_test = open(os.path.join(dataset_dir, 'ImageSets/Main/test.txt'), 'w+')
    f_train = open(os.path.join(dataset_dir, 'ImageSets/Main/train.txt'), 'w+')
    f_val = open(os.path.join(dataset_dir, 'ImageSets/Main/val.txt'), 'w+')

    for i in list:
        name = xml_anns[i][:-4] + '\n'
        if i in list_trv:
            f_trainval.write(name)
            if i in list_tr:
                f_train.write(name)
            else:
                f_val.write(name)
        else:
            f_test.write(name)

    f_trainval.close()
    f_train.close()
    f_val.close()
    f_test.close()


if __name__ == '__main__':
    _main()