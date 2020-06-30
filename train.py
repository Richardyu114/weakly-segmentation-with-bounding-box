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
# write logs
from tensorboardX import SummaryWriter
#use visdom to visualization training performance and info 
# reference:  https://blog.csdn.net/wen_fei/article/details/82979497
#import visdom 

sys.path.append('post_process/')
sys.path.append('model/')
from densecrf import run_densecrf
from fcn import FCN8s, FCN16s, FCN32s, FCNs, VGGNet
from unet import UNet
from dataset import trainloader, valloader
from update_label import update_label

class BFocalLoss(nn.Module):

    def __init__(self, gamma=2,alpha=0.5):
        super(BFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    def forward(self, inputs, targets):
        p = inputs
        loss = -self.alpha*(1-p)**self.gamma*(targets*torch.log(p+1e-12))-\
               (1-self.alpha)*p**self.gamma*((1-targets)*torch.log(1-p+1e-12))
        loss = torch.mean(loss)
        return loss


def adjust_learning_rate(optimizer, epoch_now):
        lr = 0.1
        if epoch_now >= 100:
            lr /= 10
        if epoch_now >= 150:
           lr /= 10
        for param_group in optimizer.param_groups:
               param_group['lr'] = lr

def train_model(num_epoch=30, update_epoch=10):
        # vis = visdom.Visdom()

        """train configuration"""
        device = torch.device('cuda: 3' if torch.cuda.is_available() else 'cpu')

        #backbone = VGGNet(requires_grad=True, show_params=False)
        #model_fcn = FCNs(pretrained_net=backbone, n_classes=1)
        model_unet = UNet(3, 1)
        #model = model_fcn
        model = model_unet
        model = model.to(device)
        #use pos_weight for imbalanced class
        #pos_weight > 1 will increase the recall while pos_weight < 1 will increase the precision
        #criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([0.2, 0.8]))
        #criterion = nn.BCELoss()
        criterion = BFocalLoss()
        #optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.7)
        optimizer = optim.Adam(model.parameters(), lr=1e-3) #weight_decay =1e-4
        print(model)
 
        logs_writer = SummaryWriter('logs/exp1/')
        step = 0
        print_info_step = 50

        """training"""
        for epoch in range(num_epoch):
               train_loss = 0
               # set training mode
               model.train()
               for inputs, labels in trainloader:
                      start_time = datetime.now()
                      step += 1
                      inputs = inputs.to(device)
                      labels = labels.to(device)
                      #clear gradients for next train
                      optimizer.zero_grad()
                      pred = model(inputs)
                      # 0-1 output for classification binary classes
                      pred = torch.sigmoid(pred)
                      # need to fix
                      pred = pred.view([inputs.size()[0], 1216, 1632])
                      loss = criterion(pred, labels)
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
                                            pred = pred.view([inputs.size()[0], 1216, 1632])
                                            loss = criterion(pred, labels)
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
                          torch.save(model, 'model_unet.pth')
                          print('save model done...')

               """updating label"""
               if ((epoch+1) % update_epoch == 0) and (epoch+1 != num_epoch):
                    print('waiting for updating segmentation label...')
                    update_label(model, device)
                    model.train()
                    print('updating label done, continue to train for next circle...')
               else:
                    model.train()


        logs_writer.close()
        

if __name__ == '__main__':

    train_model(num_epoch=30)





                                    













        
