#-*- coding: UTF-8 -*-
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import os
import cv2
import numpy as np
from torchvision.transforms import ToPILImage
from PIL import Image
from torchvision import transforms
from utils.utils import to_image, acc_counter,loss_counter
import ipdb


__all__ = ['Inception3', 'inception_v3']


class Inception3(nn.Module):
    
    def __init__(self, num_classes=200, args=None, threshold=0.6, transform_input=False):
        super(Inception3, self).__init__()
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3)
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288, kernel_size=3, stride=1, padding=1)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.num_classes = num_classes
        self.clas = self.classifier(num_classes)
        self.clas_fc = nn.Linear(1024,num_classes)
        self.threshold = threshold
        self._initialize_weights()
        #Optimizer
        self.loss_cross_entropy = nn.CrossEntropyLoss()


    def forward(self, x, label=None):

        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        feat1 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        x = self.Conv2d_3b_1x1(feat1)
        x = self.Conv2d_4a_3x3(x)
        feat2 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        x = self.Mixed_5b(feat2)
        x = self.Mixed_5c(x)
        feat3 = self.Mixed_5d(x)

        x = self.Mixed_6a(feat3)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        feat = self.Mixed_6e(x)
        feat = F.avg_pool2d(feat, kernel_size=3, stride=1, padding=1)
        
        
        x = self.clas[0](feat)
        x = self.clas[1](x)
        pre_logit = self.clas[2](x)
        pre_logit = pre_logit.view(pre_logit.size(0), -1)
        logits = self.clas_fc(pre_logit)
        if self.training==False:
            feature_map = x.detach().clone()
            label = torch.tensor([torch.argmax(logits,1).item()])
            label = label.long().detach().clone()
            cam_weights = self.clas_fc.weight[label]
            batchs,dims = feature_map.shape[:2]
            cams = (cam_weights.view(batchs,dims, 1, 1) *feature_map
                    ).mean(1, keepdim=False)
            self.cams = cams
            return cams,[logits,]
        return [logits,]

    def classifier(self,num_classes):
        return nn.Sequential(
            nn.Conv2d(768,1024,kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1)
        )



    def erase_feature_maps(self, atten_map_normed, feature_maps, threshold):
        # atten_map_normed = torch.unsqueeze(atten_map_normed, dim=1)
        # atten_map_normed = self.up_resize(atten_map_normed)
        if len(atten_map_normed.size())>3:
            atten_map_normed = torch.squeeze(atten_map_normed)
        atten_shape = atten_map_normed.size()

        pos = torch.ge(atten_map_normed, threshold)
        mask = torch.ones(atten_shape).cuda()
        mask[pos.data] = 0.0
        mask = torch.unsqueeze(mask, dim=1)
        #erase
        erased_feature_maps = feature_maps * Variable(mask)

        return erased_feature_maps

    def feat_select_1slide(self , cor_list , feat):
        '''input: cor_list : (b,) ; feat : (b,512,28,28)
        '''
        for i in range(feat.shape[0]): #batch_size 20
            feat_resize = self.cal_1slide(feat[i],cor_list[i])  #feat_resize: (512,28,28)
            feat_resize = feat_resize.unsqueeze(0) #(1,512,28,28)
            if i == 0 :
                feat_resize_batch = feat_resize
            else : 
                feat_resize_batch = torch.cat((feat_resize_batch , feat_resize),dim=0)
        feat_resize_batch = Variable( feat_resize_batch.cuda() ) #(b,512,28,28)

        return feat_resize_batch

    def feat_select_7slide(self ,  cor_list , feat ):
        '''input: cor_list : (b,)  ; feat : (b,512,28,28)
           output : feat_resize_batch(7,b,512,28,28) 
        '''
        for i in range(feat.shape[0]): #batch_size 20
            feat_resize = self.cal_7slide(feat[i],cor_list[i])  #feat_resize: (7,512,28,28)
            feat_resize = feat_resize.unsqueeze(1) #(7,1,512,28,28)
            if i == 0 :
                feat_resize_batch = feat_resize
            else : 
                feat_resize_batch = torch.cat((feat_resize_batch , feat_resize),dim=1)
        feat_resize_batch = Variable( feat_resize_batch.cuda() ) #(7,b,512,28,28)

        return feat_resize_batch

    def cal_size(self , feat , gt_truth ,localization_map_normed):
        '''cal_size : input:  feat shape:(b,512,28,28) , gt_truth shape:(b,)
            localization_map_normed : (b,28,28)
            output:  cordinate(b,) cordinate[i]:(x0,y0,x1,y1)
        '''
        feat = feat.detach()
        localization_map_normed = localization_map_normed.detach()
        gt_truth = gt_truth.detach()
        cor_list = self.find_highlight_region(localization_map_normed , 0.6) #corlist : (b,)
        output_list = []
        count = 0
        for i in cor_list :
            x0,y0,x1,y1 = i
            wt,ht = self.cal_wt_ht(i)
            ori_s = self.cal_score((x0,y0,x1,y1), feat[count], gt_truth[count])#x0,y0,x1,y1 all in the range of 0-27
            while wt != 0 or ht != 0:
                xt = max(x0-wt,0)
                yt = max(y0-ht,0)
                if xt == x0 and yt == y0:#if the cordinate don't change left-up is to the end
                    break
                s = self.cal_score((xt,yt,x1,y1), feat[count], gt_truth[count])
                if s>ori_s:#if new cor get higher score , refresh parameters ,expend
                    x0 , y0 = xt , yt
                    wt,ht = self.cal_wt_ht((x0,y0,x1,y1))
                    ori_s = s
                else :#if not shrink : (wt,ht)
                    wt,ht = int(1./2*wt) , int(1./2*ht)
            wt,ht = self.cal_wt_ht((x0,y0,x1,y1))
            while wt != 0 or ht != 0:
                xt,yt = min(x1+wt , 27) , min(y1+ht , 27)
                if xt == x1 and yt==y1:
                    break
                s = self.cal_score((x0,y0,xt,yt) , feat[count], gt_truth[count])
                if s>ori_s:
                    x1 , y1 = xt , yt
                    wt,ht = self.cal_wt_ht((x0 , y0 , x1, y1))
                    ori_s = s
                else:
                    wt , ht = int(1./2*wt) , int(1./2*ht)
            
            output_list.append((x0,y0,x1,y1))
        return output_list

    def cal_score(self , cor , feat_map,gt_truth):
        '''
        cor: (x0,y0,x1,y1) ; feat_map: (512,28,28) ; gt_truth: (tensor)
        the element in cor between 0-27 you should +1 in the slice operation
        '''
        x0 , y0 , x1 , y1 = cor
        feat_map = feat_map[:,y0:y1+1 , x0:x1+1]#(512,h,w)
        feat_map = feat_map.unsqueeze(1)#(512,1,h,w)
        feat_map = F.interpolate(feat_map , (28,28) ,  mode = 'bilinear',align_corners=False)
        #(512,1,28,28)
        feat_map = feat_map.squeeze(dim=1).unsqueeze(0) #(1,512,28,28)
        out_map = self.cls(feat_map) #(1,200,28,28)
        score = torch.mean( torch.mean(out_map ,dim=2),dim=2 )[0,int(gt_truth)]#the gt_label's score
        return score

    def cal_wt_ht(self , cor):
        x0 , y0 , x1 , y1 = cor
        w = int((1./2)*(x1-x0))
        h = int((1./2)*(y1-y0))
        return (w,h)

    def restore(self , heatmp , cor_list , i):
        #heatmp (batch,200,28,28) ; cor_list (list style) (batch,)

        batch = heatmp.shape[0]
        output = Variable(torch.zeros((batch,self.num_classes,28,28)).cuda())
        for idx in range(batch):            
            x0,y0,x1,y1 = self.cal_idx( cor_list[idx] , i) 
            size = (y1+1-y0,x1+1-x0 )
            heatmp_idx = heatmp[idx].unsqueeze(1) #(200,1,28,28)
            heatmp_idx =  F.interpolate(heatmp_idx , size , mode = 'bilinear',align_corners=False)
            output[idx,:,y0:y1+1,x0:x1+1] = heatmp_idx.squeeze(dim=1) #(200,w,h)
        return output #(batch,200,28,28)

    def cal_idx(self ,cordinate , i):
        x0,y0,x1,y1 = cordinate
        if i == 0:
            pass
        elif i ==1 :
            y1 = min(y1+5,27)
        elif i ==2 :
            y0 =  max(y0-5,0)
        elif i ==3 :
            x0 = max(x0-5,0)
        elif i ==4 :
            x1 = min(x1+5,27)
        elif i ==5 :
            y0 = max(y0-5,0)
            y1 = min(y1+5,27)
            x0 = max(x0-5,0)
            x1 = min(x1+5,27)
        elif i ==6 :
            y0 = max(y0-10,0)
            y1 = min(y1+10,27)
            x0 = max(x0-10,0)
            x1 = min(x1+10,27)
        else :
            raise Exception('ops Error')    
        return (x0,y0,x1,y1 )
    
    def cal_1slide(self , feature_map , cordinate):
        '''input : feature_map : (512 , 28 , 28)
           cordinate : (x0 , y0 , x1 , y1)
           output: feat_resize (512,28,28) 
        '''
        x0,y0,x1,y1 = cordinate
        cur_slide = feature_map[:,y0:y1+1,x0:x1+1]
        cur_slide = cur_slide.unsqueeze(1) #(512,1,w,h)
        feat_resize = torch.squeeze(F.interpolate(cur_slide , (28,28), mode = 'bilinear',align_corners=False))
        feat_resize = feat_resize.squeeze() #(512,28,28)
        #feat_resize = feat_resize.unsqueeze(0) #(1,512,28,28)
        return feat_resize #(512,28,28)

    def cal_7slide(self , feature_map , cordinate):
        '''input: feature_map : (512,28,28)
                  cordinate : (x0,y0,x1,y1)
           output: feat_resize_batch (7,512,28,28)
        '''
        slide_list = []
        return_list = []
        x0,y0,x1,y1 = cordinate
        #add_seven_sides
        slide_list.append(feature_map[:,y0:y1+1,x0:x1+1])
        slide_list.append(feature_map[:,y0:min(y1+6,28),x0:x1+1])
        slide_list.append(feature_map[:,max(y0-5,0):y1+1,x0:x1+1])
        slide_list.append(feature_map[:,y0:y1+1,max(x0-5,0):x1+1])
        slide_list.append(feature_map[:,y0:y1+1,x0:min(x1+6,28)])
        slide_list.append(feature_map[:,max(y0-5,0):min(y1+6,28),max(x0-5,0):min(x1+6,28)])
        slide_list.append(feature_map[:,max(y0-10,0):min(y1+11,28),max(x0-10,0):min(x1+11,28)])
        count = 0
        for cur_slide in slide_list:
            count +=1
            cur_slide = cur_slide.unsqueeze(1) #(512,1,w,h)
            cur_resize = torch.squeeze(F.interpolate(cur_slide , (28,28), mode = 'bilinear',align_corners=False))
            cur_resize = cur_resize.squeeze() #(512,28,28)
            cur_resize = cur_resize.unsqueeze(0) #(1,512,28,28)
            if count == 1 :
                feat_resize_batch = cur_resize
            else : 
                feat_resize_batch = torch.cat((feat_resize_batch , cur_resize))
        
        return feat_resize_batch #(7,512,28,28)

    def find_highlight_region(self , atten_map_normed , threshold ):
        '''
        input : atten_map_normed:(b,28,28)
        output : cor_list(b,)
        '''
        if len(atten_map_normed.size()) > 3: 
            atten_map_normed = torch.squeeze(atten_map_normed)
        #judge weather elements> 0.6
        atten_indicate_map = torch.ge(atten_map_normed , threshold) #return a bool matricx indicas whether this tensor's score bigger than threshold
        atten_indicate_indx = torch.nonzero(atten_indicate_map) #return the non zeros index  
        cor_list = []
        for i in range(atten_map_normed.shape[0]): #batch_size b
            temp = atten_indicate_indx[ atten_indicate_indx[:,0]==i ]
            y0 = torch.min(temp[:,1]).item()
            y1 = torch.max(temp[:,1]).item()
            x0 = torch.min(temp[:,2]).item()
            x1 = torch.max(temp[:,2]).item()
            cordinate = (x0,y0,x1,y1)#(0-27)
            cor_list.append(cordinate)
        return cor_list #(b,)


    def add_heatmap2img(self, img, heatmap):
        # assert np.shape(img)[:3] == np.shape(heatmap)[:3]

        heatmap = heatmap* 255
        color_map = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        img_res = cv2.addWeighted(img.astype(np.uint8), 0.5, color_map.astype(np.uint8), 0.5, 0)

        return img_res

    #because it cals with batch , so the deminson is two
    #shape[0] stands for the batch size
    def get_loss(self, logits, gt_labels,current_epoch):
        gt = gt_labels.long()
        loss_cls = self.loss_cross_entropy(logits[0], gt)
        return [loss_cls, ]

    def get_localization_maps(self):
        #map1 = self.normalize_atten_maps(self.map1)
        map_erase = self.normalize_atten_maps(self.map_erase) #only return map_erase
        return map_erase
        # return map_erase

    def get_heatmaps(self, gt_label):
        map1 = self.get_atten_map(self.map1, gt_label)
        return [map1,]


    def get_maps(self, gt_label):
        map1 = self.get_atten_map(self.map1, gt_label)
        return [map1, ]


    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed



    def get_atten_map(self, feature_maps, gt_labels, normalize=True):
        label = gt_labels.long()  #lable.shape : (20,)

        feature_map_size = feature_maps.size()  #feature_map_size : (20,200,28,28)
        batch_size = feature_map_size[0] #batch_size : 20

        atten_map = torch.zeros([feature_map_size[0], feature_map_size[2], feature_map_size[3]]) 
        atten_map = Variable(atten_map.cuda()) #atten_map : (20,28,28)
        for batch_idx in range(batch_size):
            atten_map[batch_idx,:,:] = torch.squeeze(feature_maps[batch_idx, label.data[batch_idx], :,:]) #fetch the 'label'th map ,is the atten_map

        if normalize:
            atten_map = self.normalize_atten_maps(atten_map)

        return atten_map


    def loss_init(self):
        self.loss_ct = loss_counter()
        self.acc_count = acc_counter()
    
    def loss_count(self,logits,cls_pre,clas):
        self.loss_ct.count(logits[0])
        self.acc_count.count(cls_pre[:clas.shape[0],:].argmax(1).cpu().numpy(),clas.cpu().numpy())

    def loss_reset(self):
        self.loss_ct.reset()
        self.acc_count.reset()

    def loss_output(self):
        output = 'loss:{:.4f},cls:{:.4f}'\
            .format(self.loss_ct(),self.acc_count())
        return output

#初始化权重
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class inference(nn.Module):
    def __init__(self , num_classes=200):
        super(inference, self).__init__() #可以继承nn.Module里的属性，主要是train属性吧
        self.cls_fc6 = nn.Sequential(
            nn.Conv2d(768, 1024, kernel_size=3, padding=1, dilation=1),   #fc6
            nn.ReLU(True),
        )
        self.cls_fc7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),   #fc6
            nn.ReLU(True))
        self.cls_fc8 = nn.Conv2d(1024, num_classes, kernel_size=1, padding=0)  #fc8

    def forward(self , x):

        if self.training:
            x = F.dropout(x, 0.5)
        x = self.cls_fc6(x)

        if self.training:
            x = F.dropout(x, 0.5)
        x = self.cls_fc7(x)

        if self.training:
            x = F.dropout(x, 0.5)
        out1 = self.cls_fc8(x)

        return out1


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)

class InceptionB(nn.Module):

    def __init__(self, in_channels, kernel_size=3, stride=2, padding=0):
        self.stride = stride
        super(InceptionB, self).__init__()
        self.branch3x3 = BasicConv2d(in_channels, 384,
                                     kernel_size=kernel_size, stride=stride, padding=padding)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=3, stride=stride, padding=padding)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels):
        super(InceptionD, self).__init__()
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=3, stride=2)

    def forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels):
        super(InceptionE, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1(x)
        # 1 x 1 x 768
        x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

def model(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    """
    model = Inception3(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model
