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
import ipdb


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes=1000, args=None , threshold=0.6):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
        

        self.num_classes = args.num_classes
        self.cls = inference(self.num_classes)
        self.cls_erase = inference(self.num_classes)
        
        self._initialize_weights()
        self.onehot = args.onehot
        self.max_weights = 1
        self.threshold = threshold

        #Optimizer
        self.loss_cross_entropy = nn.CrossEntropyLoss()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, label=None): 
        self.img_erased = x
        #Backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        x = self.layer4(feat3)

        feat = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)#feat:(b,512,28,28)
        # Branch A      
        out = self.cls(feat)
        self.map1 = out
        logits_1 = torch.mean(torch.mean(out, dim=2), dim=2)
        
        out_erase = self.cls_erase(feat)#out_erase:(b,200,28,28)
        self.map_erase = out_erase
        logits_ers = torch.mean(torch.mean(out_erase, dim=2), dim=2)#logits_ers:(b,200)
        return [logits_1, logits_ers]
    
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
    def get_loss(self, logits, gt_labels):
        if self.onehot == 'True':
            gt = gt_labels.float()
        else:
            gt = gt_labels.long()
        loss_cls = self.loss_cross_entropy(logits[0], gt)
        loss_cls_ers = self.loss_cross_entropy(logits[1], gt)
        
        booltensor = torch.zeros( (logits[0].shape[0],logits[0].shape[1]) ).type(torch.uint8)
        for i in range(gt.shape[0]):
            booltensor[i][gt[i]]=1
        lg0 = logits[0][booltensor]
        lgerror = logits[1][booltensor]
        stand = Variable(torch.zeros(logits[0].shape[0],1).cuda()).squeeze()
        lossmax = torch.max(lg0-lgerror,stand)
        lossmax = torch.sum(lossmax)/lossmax.shape[0] #average it
        loss_val = loss_cls + loss_cls_ers + self.max_weights*lossmax

        return [loss_val, ]

    def get_localization_maps(self):
        #map1 = self.normalize_atten_maps(self.map1)
        map_erase = self.normalize_atten_maps(self.map_erase) #only return map_erase
        map1 = self.normalize_atten_maps(self.map1)
        #cut_heatmp=[]
        #for i in range(4):
        #    cut_heatmp.append(self.normalize_atten_maps(self.cut_heatmap[i]))
        #
        information_dict = map_erase
        return information_dict   
        # return map_erase

    def get_heatmaps(self, gt_label):
        map1 = self.get_atten_map(self.map1, gt_label)
        return [map1,]

    def get_fused_heatmap(self, gt_label):
        maps = self.get_heatmaps(gt_label=gt_label)
        fuse_atten = maps[0]
        return fuse_atten

    def get_maps(self, gt_label):
        map1 = self.get_atten_map(self.map1, gt_label)
        return [map1, ]

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

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

    def save_erased_img(self, img_path, img_batch=None):
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
        if img_batch is None:
            img_batch = self.img_erased
        if len(img_batch.size()) == 4:
            batch_size = img_batch.size()[0]
            for batch_idx in range(batch_size):
                imgname = img_path[batch_idx]
                nameid = imgname.strip().split('/')[-1].strip().split('.')[0]

                # atten_map = F.upsample(self.attention.unsqueeze(dim=1), (321,321), mode='bilinear')
                atten_map = F.upsample(self.attention.unsqueeze(dim=1), (224,224), mode='bilinear')
                # atten_map = F.upsample(self.attention, (224,224), mode='bilinear')
                # mask = F.sigmoid(20*(atten_map-0.5))
                mask = atten_map
                mask = mask.squeeze().cpu().data.numpy()

                img_dat = img_batch[batch_idx]
                img_dat = img_dat.cpu().data.numpy().transpose((1,2,0))
                img_dat = (img_dat*std_vals + mean_vals)*255

                mask = cv2.resize(mask, (321,321))
                img_dat = self.add_heatmap2img(img_dat, mask)
                save_path = os.path.join('../save_bins/', nameid+'.png')
                cv2.imwrite(save_path, img_dat)

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
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1, dilation=1),   #fc6
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


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def model(pretrained=False, threshold=None, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model
