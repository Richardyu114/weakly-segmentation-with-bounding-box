"""
load the model and dataset to train the model

Based on:
   https://github.com/bat67/pytorch-FCN-easiest-demo

TODO:
    1.select different backbones and segmentation models
    2.at given epoch, use densecrf to update segmentation labels
    3.better metric info to display when training model--need to think
    4.use visdom to visualization locally
    5.better loss function(balance positive-negative pixel object and big-small object)
      average the negative label loss and sum the whole positive loss
    6.how to deal with ignore regions label
    7.using pos_weight
    8.use the val dataset better, not just for avoiding overfitting 
    9.self attention
    10.will GAN help?
----------------------------------------------------------------------------------------
After updating the segmentation labels on my dataset, the performance is worse than 
not updating...
However, it may be resulted from the BCELoss(can not handle the imbalanced sample).
----------------------------------------------------------------------------------------

"""
# for run time info
from datetime import datetime
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import _Loss, _WeightedLoss
# write logs
from tensorboardX import SummaryWriter
#use visdom to visualization training performance and info 
# reference:  https://blog.csdn.net/wen_fei/article/details/82979497
#import visdom 

sys.path.append('post_process/')
sys.path.append('model/')
from densecrf import run_densecrf

#from fcn import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
#from fcn_2 import FCN8
#from unet import UNet,UNetResnet
from deeplabv3_plus import DeepLab
#from pspnet import PSPNet, PSPDenseNet

from dataset import trainloader, valloader
#from update_label import update_label

import lovasz_losses as L  #lovasz loss

## focal loss
class BFocalLoss(nn.Module):

    def __init__(self, gamma=0,alpha=0.95):
        super(BFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, inputs, targets):
        p = inputs
        loss = -self.alpha*(1-p)**self.gamma*(targets*torch.log(p+1e-12))-\
               (1-self.alpha)*p**self.gamma*((1-targets)*torch.log(1-p+1e-12))
        loss = torch.mean(loss)
        return loss

# bce with label smooth
# https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks/blob/master/utils/losses/loss.py
class CrossEntropyLoss2dLabelSmooth(nn.Module):
    """
    Refer from https://arxiv.org/pdf/1512.00567.pdf
    :param target: N,
    :param n_classes: int
    :param eta: float
    :return:
        N x C onehot smoothed vector
    """

    def __init__(self, weight=None, epsilon=0.01, reduction='mean'):
        super(CrossEntropyLoss2dLabelSmooth, self).__init__()
        self.epsilon = epsilon
        #self.nll_loss = nn.CrossEntropyLoss(weight, ignore_index=ignore_label, reduction=reduction)
        self.loss = nn.BCELoss(weight=weight,reduction=reduction)

    def forward(self, output, target):
        """
        Forward pass
        :param output: torch.tensor (NxC), raw logits
        :param target: torch.tensor (N)
        :return: scalar
        """
        #n_classes = output.size(1)
        n_classes  = 1
        # batchsize, num_class = input.size()
        # log_probs = F.log_softmax(inputs, dim=1)
        #targets = torch.zeros_like(output).scatter_(1, target.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * target + self.epsilon / n_classes

        #return self.nll_loss(output, targets)
        return self.loss(output,targets)
 
## dice loss
# https://github.com/yassouali/pytorch_segmentation/blob/master/utils/losses.py
class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, output, target):
        # output_flat = output.contiguous().view(-1)
        # target_flat = target.contiguous().view(-1)
        intersection = (output * target).sum(dim=[1,2])
        loss = 1 - ((2 * intersection + self.smooth) /
                    (output.sum(dim=[1,2]) + target.sum(dim=[1,2]) + self.smooth))
        return loss.mean()

class FL_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', weight=None,combination_w = 10):
        super(FL_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        #self.cross_entropy = nn.BCELoss(weight,reduction=reduction)
        self.focal_loss = BFocalLoss()
        # 两个loss之间的权重
        # focal loss的值一般比较小，可能加在一起会被隐藏
        self.combination_w = combination_w
    
    def forward(self, output, target):
        #CE_loss = self.cross_entropy(output, target)
        fl_loss = self.focal_loss(output,target)
        dice_loss = self.dice(output, target)
        return self.combination_w * fl_loss +  dice_loss

#### mixup
# https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py
def mixup_data(in_img, in_label, alpha=0.6):
    #alpha in [0.1,0.4] in paper has better gain(for imagenet)
    #for cifar-10 is 1.
    if alpha > 0:
       lam = np.random.beta(alpha, alpha)
    else:
       lam = 1
    
    Batch_Size = in_img.size()[0]
    Index = torch.randperm(Batch_Size)
    mixed_x = lam * in_img + (1 - lam) * in_img[Index, :]
    y_a, y_b = in_label, in_label[Index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



def adjust_learning_rate(optimizer, epoch_now):
        lr = 0.1
        if epoch_now >= 100:
            lr /= 10
        if epoch_now >= 150:
           lr /= 10
        for param_group in optimizer.param_groups:
               param_group['lr'] = lr

def train_model(num_epoch=30):#, update_epoch=10):
        # vis = visdom.Visdom()

        """train configuration"""
        device = torch.device('cuda: 1' if torch.cuda.is_available() else 'cpu')
        #net=torch.nn.DataParallel(model,device_ids=[0,1])


        # backbone = VGGNet(requires_grad=True, show_params=False)
        # model_fcn = FCNs(pretrained_net=backbone, n_classes=1)
        # # model_fcn = FCN8(1)
        # model = model_fcn
        
        #model_unet = UNet(1)
        #model_unet = UNetResnet(1)
        #model = model_unet

        #model_deeplab = DeepLab(1,backbone='xception')
        model_deeplab = DeepLab(1,backbone='resnet')
        model = model_deeplab
        
        # model_psp = PSPNet(1)
        #model = model_psp


        model = model.to(device)
        #use pos_weight for imbalanced class
        #pos_weight > 1 will increase the recall while pos_weight < 1 will increase the precision
        #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([0.2, 0.8]))
        #criterion = nn.BCELoss()
        #criterion = BFocalLoss()
        #criterion = CrossEntropyLoss2dLabelSmooth()
        #criterion =  DiceLoss()
        criterion = FL_DiceLoss(combination_w=100)

        #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9,weight_decay = 0.0001)
        optimizer = optim.Adam(model.parameters(), lr=1e-3) #1e-3 weight_decay =1e-4
        print(model)
 
        logs_writer = SummaryWriter('logs/')
        step = 0
        print_info_step = 50

        """training"""
        for epoch in range(num_epoch):
            #    if epoch > 0: 
            #       torch.save(model, 'model_epoch' + str(epoch) + '.pth')
            #       print('save model_epoch' + str(epoch) + ' done...')
               train_loss = 0
               # set training mode
               model.train()
               for inputs, labels in trainloader:
                      start_time = datetime.now()
                      step += 1
                      inputs = inputs.to(device)
                      labels = labels.to(device)

                      # if use mixup
                      inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)

                      #clear gradients for next train
                      optimizer.zero_grad()
                      pred = model(inputs)
                      # 0-1 output for classification binary classes
                      
                      
                      pred = torch.sigmoid(pred)
                      # need to fix
                      # drone data -1024x1024
                      pred = pred.view([inputs.size()[0], 1024, 1024])
                    
                      #loss = criterion(pred, labels)

                      # if use mixup
                      loss = mixup_criterion(criterion,pred,labels_a,labels_b,lam)

                      #pred为logits
                      #loss = L.lovasz_hinge(pred,labels,per_image=False)
                      # 二分类lovasz-sigmoid损失函数
                      #loss = L.lovasz_softmax(pred, labels, classes=[1])#, ignore=2)
                      
                      loss.backward()
                      train_loss += loss.item()
                      #apply gradient
                      optimizer.step()

                      if step % print_info_step == 0:
                          val_loss = 0
                          model.eval()
                          with torch.no_grad():
                                     for inputs, labels in valloader:
                                            inputs = inputs.to(device)
                                            labels = labels.to(device)
                                            # clear gradients
                                            optimizer.zero_grad()
                                            pred = model(inputs)
                                            

                                            pred = torch.sigmoid(pred)
                                            # need to fix
                                            # drone data- 1024x1024
                                            pred = pred.view([inputs.size()[0], 1024, 1024])
                                            #loss = criterion(pred, labels)
                                            
                                            loss = criterion(pred, labels)

                                            #loss = L.lovasz_hinge(pred,labels,per_image=False)
                                            # 二分类lovasz-sigmoid损失函数
                                            #loss = L.lovasz_softmax(pred, labels, classes=[1])#, ignore=2)
                                            val_loss += loss.item()                                          

                          end_time =datetime.now()
                          run_time = (end_time - start_time).seconds
                          # write loss info into logs file
                          logs_writer.add_scalar('Train Loss', train_loss/print_info_step, global_step = step)
                          logs_writer.add_scalar('Val Loss', val_loss/len(valloader), global_step = step)
                          # print info on terminal
                          print('INFO: Epoch: {}({})...Train Loss: {}...Val Loss: {}...Running Time: {}s'.format(
                                        epoch+1,  num_epoch, '%.4f' %(train_loss / print_info_step), '%.4f' %(
                                        val_loss / len(valloader)), run_time ))

                          train_loss = 0     
                          model.train()                   
                          #torch.save(model, 'model_fcn.pth')
                          #torch.save(model, 'model.pth')
                          #print('save model done...')

               torch.save(model, 'model_epoch' + str(epoch+1) + '.pth')
               print('save model_epoch' + str(epoch+1) + ' done...')

               """updating label"""
            #    if ((epoch+1) % update_epoch == 0) and (epoch+1 != num_epoch):
            #         print('waiting for updating segmentation label...')
            #         update_label(model, device)
            #         model.train()
            #         print('updating label done, continue to train for next circle...')
            #    else:
            #         model.train()


        logs_writer.close()
        

if __name__ == '__main__':

    train_model(num_epoch=10)





                                    













        
