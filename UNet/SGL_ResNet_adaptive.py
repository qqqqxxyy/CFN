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
import sys
import json
import time
import random
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
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
        self.num_classes = num_classes
        self.cls_a = self.classifier(512*4, num_classes)
        self.cls_b = self.classifier(512*4, num_classes)
        
        self._initialize_weights()
        self.onehot = False
        self.max_weights = 1
        self.threshold = [0.6]
        self.adaptive_version = int(1)
        self.resolution=int(128/8)
        self.args = args
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

    def forward(self, x, label=None, return_cam=False): 
        '''cam返回方式时直接featmap传给classifierB，不经过增强
        的指令设置：
        return_cam == True and adaptive_version = -1
        '''
        #t1 = time.time()
        self.img_erased = x
        #Backbone
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        print('feat3_shape: {}'.format(feat3.shape))
        x = self.layer4(feat3)
        print('x_shape: {}'.format(x.shape))
        #backbone 提取 featuremap feat_map(b,2048,28,28)
        feat_map = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        logit_a, ht_ftmaps_a = self.clss_process(feat_map,'a')
        
        if self.training==False :
            # if self.args.tencrop=='True':
            #     return [logit_a,],ht_ftmaps_a

            ht_ftmaps_b = self.clss_process(feat_map,'b_1')
            CAM_b = self.cams(ht_ftmaps_b,torch.tensor(1),'b',normalize=True)   
            #self.args.scg == True
            # if True: 
            #     sc_3, sc_3_so = self.hsc(feat3, fo_th=self.args.scg_fosc_th,
            #                             so_th=self.args.scg_sosc_th, order=self.args.scg_order)
            #     sc_4, sc_4_so = self.hsc(x, fo_th=self.args.scg_fosc_th,
            #                             so_th=self.args.scg_sosc_th, order=self.args.scg_order)
            #     print('sc_3_shape: {}'.format(sc_3.shape))
            #     print('sc_4_shape: {}'.format(sc_4.shape))
            #     print('CAM_b_shape: {}'.format(CAM_b.shape))
            #     ipdb.set_trace()
            #     # sc_3, sc_3_so = self.hsc(feat3, fo_th=self.args.scg_fosc_th,
            #     #                         so_th=self.args.scg_sosc_th, order=self.args.scg_order)
            #     # sc_4, sc_4_so = self.hsc(x, fo_th=self.args.scg_fosc_th,
            #     #                         so_th=self.args.scg_sosc_th, order=self.args.scg_order)
            #     CAM_scg = self.cam_scg(CAM_b, (sc_3,sc_4),(sc_3_so,sc_4_so))
            #     return [logit_a],CAM_scg
            # else:

            return [logit_a],CAM_b.detach().cpu().numpy()
        else: 
            # Branch A 
            #t2 = time.time() 
            htmp_a = self.cams(ht_ftmaps_a,label,'a',normalize=True)          
            #得到局部特征块的坐标列表 cor_list
            #t3 = time.time()
            cor_list = self.cal_size(feat_map,label,htmp_a,version=self.adaptive_version)
            #t4 = time.time()
            #局部特征块
            feat_resize = self.feat_select_7slide(cor_list,feat_map)
            for i in range(7):
                #loc_ftmp_b局部空间特征 (b,1024,28,28)
                loc_ftmp_b = self.clss_process(feat_resize[i],'b_1')
                #feat_resized 插值之后的空间特征（b,1024,28,28）
                feat_restored = self.restore(loc_ftmp_b,cor_list,i)
                if i==0:
                    ftmp_b = feat_restored
                else:
                    ftmp_b = torch.max(ftmp_b,feat_restored)
            self.htmp_b = self.cams(ftmp_b,label,'b',normalize=True)
            #t5 = time.time()
            #self.time_test([t2-t1,t3-t2,t4-t3,t5-t4])
            if self.training == False:
                return [logit_a],self.htmp_b
            else:
                logit_b = self.clss_process(ftmp_b,'b_2')
                return [logit_a,logit_b]

    def time_test(self,time_list):
        paths = os.path.join('/home/zmwang/project/qxy/DJL/save_bins','time.txt')
        with open(paths,'a') as f:
            json.dump(time_list,f)
            f.write('\n')

    def cam_scg(self,CAM,sc_fo,sc_so):
        '''CAM: (1,14,14) (b,w,h)
        sc_fo : (sc_4,sc_5)
        sc_so : (sc_4_so, sc_5_so)
        (sc_4,sc_5),(sc_4_so,sc_5_so): (1,196,196) (b,w*h,w*h)
        from first order and second order get HSC map (b,w*h,w*h)
        '''
        fg_th , bg_th = self.args.scg_fg_th, self.args.scg_bg_th
        cam_map_cls = CAM
        sc_map = self.get_hsc(sc_fo,sc_so)

        sc_map = sc_map.squeeze().data.cpu().numpy() #196，196
        cam_map_cls = cam_map_cls.squeeze().data.cpu().numpy() #14,14
        cam_map_cls_vector = cam_map_cls.reshape(-1)
        wh_sc = sc_map.shape[0]
        h_sc, w_sc = int(np.sqrt(wh_sc)), int(np.sqrt(wh_sc))

        #positive
        cam_map_cls_id = np.arange(wh_sc).astype(np.int)
        cam_map_cls_th_ind_pos = cam_map_cls_id[cam_map_cls_vector >= fg_th]
        sc_map_sel_pos = sc_map[:,cam_map_cls_th_ind_pos]
        #归一化
        sc_map_sel_pos = (sc_map_sel_pos - np.min(sc_map_sel_pos,axis=0, keepdims=True))/(
                np.max(sc_map_sel_pos, axis=0, keepdims=True) - np.min(sc_map_sel_pos, axis=0, keepdims=True) + 1e-10)
        # cam_map_cls_val_pos = cam_map_cls_vector[cam_map_cls_th_ind_pos].reshape(1,-1)
        # aff_map_sel_pos = np.sum(aff_map_sel_pos * cam_map_cls_val_pos, axis=1).reshape(h_aff, w_aff)
        #取平均
        if sc_map_sel_pos.shape[1] > 0:
            sc_map_sel_pos = np.sum(sc_map_sel_pos, axis=1).reshape(h_sc, w_sc)
            sc_map_sel_pos = (sc_map_sel_pos - np.min(sc_map_sel_pos))/( np.max(sc_map_sel_pos) - np.min(sc_map_sel_pos)  + 1e-10)
        else:
            sc_map_sel_pos = 0

        #negtive
        #完全重复一篇positiv的流程，只是>=fg_th 变成了<=bg_th
        cam_map_cls_th_ind_neg = cam_map_cls_id[cam_map_cls_vector <= bg_th]

        sc_map_sel_neg = sc_map[:, cam_map_cls_th_ind_neg]
        sc_map_sel_neg = (sc_map_sel_neg - np.min(sc_map_sel_neg,axis=0, keepdims=True))/(
                np.max(sc_map_sel_neg, axis=0, keepdims=True) - np.min(sc_map_sel_neg, axis=0, keepdims=True)+ 1e-10)
        # cam_map_cls_val_neg = cam_map_cls_vector[cam_map_cls_th_ind_neg].reshape(1, -1)
        # aff_map_sel_neg = np.sum(aff_map_sel_neg * (1-cam_map_cls_val_neg), axis=1).reshape(h_aff, w_aff)
        if sc_map_sel_neg.shape[1] > 0:
            sc_map_sel_neg = np.sum(sc_map_sel_neg, axis=1).reshape(h_sc, w_sc)
            sc_map_sel_neg = (sc_map_sel_neg - np.min(sc_map_sel_neg))/(np.max(sc_map_sel_neg)-np.min(sc_map_sel_neg) + 1e-10)
        else:
            sc_map_sel_neg = 0
        sc_map_cls_i = sc_map_sel_pos - sc_map_sel_neg
        #取>0的部分，归一化
        sc_map_cls_i = sc_map_cls_i * (sc_map_cls_i>=0)
        sc_map_cls_i = (sc_map_cls_i-np.min(sc_map_cls_i))/(np.max(sc_map_cls_i) - np.min(sc_map_cls_i)+1e-10)
        #最后再取最大值
        sc_map_cls= np.expand_dims(sc_map_cls_i,0) #修正之后的sc_map #14*14
        return sc_map_cls   

    def get_hsc(self,sc_fo,sc_so):
        '''input:
        sc_fo : (sc_4,sc_5)
        sc_so : (sc_4_so, sc_5_so)
        (sc_4,sc_5),(sc_4_so,sc_5_so): (1,196,196) (b,w*h,w*h)
        output:
        HSC: (1,196,196) (b,w*h,w*h)
        '''
        sc_maps = []
        for  sc_map_fo_i, sc_map_so_i in zip(sc_fo,sc_so):
            sc_map_i = torch.max(sc_map_fo_i, self.args.scg_so_weight * sc_map_so_i)
            sc_map_i = sc_map_i / (torch.sum(sc_map_i, dim=1, keepdim=True) + 1e-10)
            sc_maps.append(sc_map_i)
        HSC = sc_maps[-1]#+0.5*sc_maps[-2]
        return HSC

    def hsc(self, f_phi, fo_th=0.1, so_th=0.1, order=2):
        """
        Calculate affinity matrix and update feature.
        :param feat:
        :param f_phi:
        :param fo_th:
        :param so_weight:t_trace()
        :return:
        """
        n, c_nl, h, w = f_phi.size()
        c_nl = f_phi.size(1)
        f_phi = f_phi.permute(0, 2, 3, 1).contiguous().view(n, -1, c_nl)
        f_phi_normed = f_phi/(torch.norm(f_phi, dim=2, keepdim=True)+1e-10) #对channel取模

        # first order
        non_local_cos = F.relu(torch.matmul(f_phi_normed, f_phi_normed.transpose(1,2)))
        non_local_cos[non_local_cos<fo_th] = 0
        non_local_cos_fo = non_local_cos.clone()
        non_local_cos_fo = non_local_cos_fo / (torch.sum(non_local_cos_fo, dim=1, keepdim=True) + 1e-5)

        # high order
        base_th = 1./(h*w)
        non_local_cos[:, torch.arange(h * w), torch.arange(w * h)] = 0
        non_local_cos = non_local_cos/(torch.sum(non_local_cos,dim=1, keepdim=True) + 1e-5)
        non_local_cos_ho = non_local_cos.clone()
        so_th = base_th * so_th
        # ipdb.set_trace()
        for _ in range(order-1):
            non_local_cos_ho = torch.matmul(non_local_cos_ho, non_local_cos)
            # non_local_cos_ho[:, torch.arange(h * w), torch.arange(w * h)] = 0
            non_local_cos_ho = non_local_cos_ho / (torch.sum(non_local_cos_ho, dim=1, keepdim=True) + 1e-10)
        # non_local_cos_ho = non_local_cos_ho - torch.min(non_local_cos_ho, dim=1, keepdim=True)[0]
        #non_local_cos_ho = non_local_cos_ho / (torch.max(non_local_cos_ho, dim=1, keepdim=True)[0] + 1e-10)
        non_local_cos_ho[non_local_cos_ho < so_th] = 0
        return  non_local_cos_fo, non_local_cos_ho

    def cams(self, ht_ftmaps, label,branch_mark,normalize=True):
        '''input：
        ht_ftmaps:(b,1024,28,28) #权重文件
        branch_mark:'a'or'b'
        normalize:决定是否归一化
        output:
        cams:(b,28,28) #热力图
        '''
        ftmp = ht_ftmaps.detach().clone()
        label = label.long().detach().clone()
        cam_weigths = eval('self.cls_'+branch_mark)[3].weight[label]
        batchs,dims = ftmp.shape[:2]
        cams = (cam_weigths.view(batchs,dims,1,1)*ftmp)\
                .mean(1,keepdim=False)

        if normalize:
            cams = self.normalize_atten_maps(cams)
        return cams

    def clss_process(self,feat_map,branch_mark):
        '''forward pass progress of CAM,
        综合了两个分类器分支的训练过程
        input: feat_map :(b,28,28)
        branch_mark: 'a' or 'b'
        output: ht_ftmaps:(b,1024,28,28) 权重文件
        x:(b,200) 预测概率

        '''
        if branch_mark=='a':
            ht_ftmaps = self.cls_a[:2](feat_map)
            x = self.cls_a[2](ht_ftmaps)
            x = x.view(x.size(0), -1)
            x = self.cls_a[3](x)
            return x,ht_ftmaps
        if branch_mark=='b_1':
            ht_ftmaps = self.cls_b[:2](feat_map)
            return ht_ftmaps
        if branch_mark == 'b_2':
            x = self.cls_b[2](feat_map)
            x = x.view(x.size(0), -1)
            x = self.cls_b[3](x)
            return x

    def classifier(self, in_planes, out_planes):
        return nn.Sequential(
            # nn.Dropout(0.5),
            nn.Conv2d(in_planes, 1024, kernel_size=3, padding=1),   #fc6
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(1024,out_planes)
        )



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

    def cal_size(self , feat , gt_truth ,localization_map_normed,version=0):
        '''input:  feat shape:(b,512,28,28) , gt_truth shape:(b,)
            localization_map_normed : (b,28,28)
            output:  cordinate(b,) cordinate[i]:(x0,y0,x1,y1)
            version:[0,1,2],v0:左上右下两次滑动，就是修改之前的SGL_CAM
            v1：左上，右上，左下，右下四次滑动取x0,y0,x1,y1的最大值,中心不变
            v2：左上，右上，右下，左下四次滑动，中心改变
        '''
        #t1 = time.time()
        reso = self.resolution-1
        feat = feat.detach()
        localization_map_normed = localization_map_normed.detach()
        gt_truth = gt_truth.detach()
        #通过热力图(loc_map_normed)得到初始坐标列表cor_list（b,）
        cor_list = self.find_highlight_region(localization_map_normed , 0.6) #corlist : (b,)
        #cor_list = self.find_highlight_region2(localization_map_normed , 0.6)
        #t2 = time.time()
        output_list = []
        count = -1
        if version ==0 :
            for i in cor_list :
                count+=1
                x0,y0,x1,y1 = i
                wt,ht = self.cal_wt_ht(i)#通过图片的坐标计算出宽和高
                #计算对应图片初始坐标下的得分
                ori_s = self.cal_score((x0,y0,x1,y1), feat[count], gt_truth[count])#x0,y0,x1,y1 all in the range of 0-27
                #向左上方滑动
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
                    xt,yt = min(x1+wt , reso) , min(y1+ht , reso)
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
        
        elif version == 1:
            for cordinate in cor_list:
                count+=1
                x0,y0,x1,y1 = cordinate
                #print(cordinate)
                #初始化 #向左上方滑动
                ori_ss = self.cal_score((x0,y0,x1,y1),feat[count],gt_truth[count])
                wtt,htt = self.cal_wt_ht(cordinate)
                ori_s = ori_ss
                wt,ht = wtt,htt
                xul , yul = x0 , y0
                while wt != 0 or ht != 0:
                    xt ,yt = max(xul-wt,0), max(yul-ht,0)
                    if xt == xul and yt == yul: #如果不能再往左上滑动了
                        break
                    s = self.cal_score( (xt,yt,x1,y1),feat[count],gt_truth[count] )
                    if s > ori_s: #如果扩展后的图像分类得分更高的话,扩展坐标位置,更新宽、高、得分
                        xul,yul = xt,yt
                        wt,ht = self.cal_wt_ht((xt,yt,x1,y1))
                        ori_s = s
                    else : #如果得分更低的话，就收缩扩展框，直到满足两个停机条件为止
                        wt,ht = int(1./2*wt),int(1./2*ht)

                #初始化 #向右上方滑动
                ori_s = ori_ss
                wt,ht =  wtt,htt
                #print(wt,ht)                    
                xur,yur=x1,y0
                while wt !=0 or ht != 0 :
                    xt,yt = min(xur+wt,reso),max(yur-ht,0)
                    if xt == xur and yt == yur:
                        break
                    s = self.cal_score( (x0,yt,xt,y1),feat[count],gt_truth[count] )
                    if s> ori_s:
                        xur,yur = xt,yt
                        wt,ht = self.cal_wt_ht((x0,yt,xt,y1))
                        ori_s = s
                    else :
                        wt,ht = int(1./2*wt),int(1./2*ht)

                #初始化 #向右下方滑动
                ori_s = ori_ss
                wt,ht =  wtt,htt                    
                xlr,ylr=x1,y1   
                while wt != 0 or ht != 0:
                    xt,yt = min(xlr+wt,reso),min(ylr+ht,reso)
                    if xt == xlr and yt == ylr:
                        break
                    s = self.cal_score((x0,y0,xt,yt),feat[count],gt_truth[count] )
                    if s> ori_s:
                        xlr,ylr = xt,yt
                        wt,ht = self.cal_wt_ht((x0,y0,xt,yt))
                        ori_s = s
                    else:
                        wt,ht = int(1./2*wt),int(1./2*ht) 
                
                #初始化 #向左下方滑动
                ori_s = ori_ss
                wt,ht =  wtt,htt                      
                xll,yll=x0,y1   
                while wt != 0 or ht != 0:
                    xt,yt = max(xll-wt,0),min(yll+ht,reso)
                    if xt == xll and yt == yll:
                        break
                    s = self.cal_score((xt,y0,x1,yt),feat[count],gt_truth[count] )
                    if s> ori_s:
                        xll,yll = xt,yt
                        wt,ht = self.cal_wt_ht((xt,y0,x1,yt))
                        ori_s = s
                    else:
                        wt,ht = int(1./2*wt),int(1./2*ht)         
                
                x0,y0,x1,y1 = min(xul,xll),min(yul,yur),max(xur,xlr),max(ylr,yll)
                output_list.append((x0,y0,x1,y1))
        elif version ==2:
            '''第二个版本，四个方向滑动并且改变中心方向
            '''
            for cordinate in cor_list:
                count+=1
                x0,y0,x1,y1 = cordinate
                #print(cordinate)
                #初始化 #向左上方滑动
                ori_ss = self.cal_score((x0,y0,x1,y1),feat[count],gt_truth[count])
                wtt,htt = self.cal_wt_ht(cordinate)
                ori_s = ori_ss
                wt,ht = wtt,htt
                xul , yul = x0 , y0
                while wt != 0 or ht != 0:
                    xt ,yt = max(xul-wt,0), max(yul-ht,0)
                    if xt == xul and yt == yul: #如果不能再往左上滑动了
                        break
                    s = self.cal_score( (xt,yt,x1,y1),feat[count],gt_truth[count] )
                    if s > ori_s: #如果扩展后的图像分类得分更高的话,扩展坐标位置,更新宽、高、得分
                        xul,yul = xt,yt
                        wt,ht = self.cal_wt_ht((xul,yul,x1,y1))
                        ori_s = s
                    else : #如果得分更低的话，就收缩扩展框，直到满足两个停机条件为止
                        wt,ht = int(1./2*wt),int(1./2*ht)
                x0,y0 = xul,yul #更新坐标
                #初始化 #向右上方滑动
                ori_s = ori_ss
                wt,ht =  wtt,htt                    
                xur,yur=x1,y0
                while wt !=0 or ht != 0 :
                    xt,yt = min(xur+wt,reso),max(yur-ht,0)
                    if xt == xur and yt == yur:
                        break
                    s = self.cal_score( (x0,yt,xt,y1),feat[count],gt_truth[count] )
                    if s> ori_s:
                        xur,yur = xt,yt
                        wt,ht = self.cal_wt_ht((x0,yt,xt,y1))
                        ori_s = s
                    else :
                        wt,ht = int(1./2*wt),int(1./2*ht)
                x1,y0 = xur,yur #更新坐标
                #初始化 #向右下方滑动
                ori_s = ori_ss
                wt,ht =  wtt,htt                    
                xlr,ylr=x1,y1   
                while wt != 0 or ht != 0:
                    xt,yt = min(xlr+wt,reso),min(ylr+ht,reso)
                    if xt == xlr and yt == ylr:
                        break
                    s = self.cal_score((x0,y0,xt,yt),feat[count],gt_truth[count] )
                    if s> ori_s:
                        xlr,ylr = xt,yt
                        wt,ht = self.cal_wt_ht((x0,y0,xt,yt))
                        ori_s = s
                    else:
                        wt,ht = int(1./2*wt),int(1./2*ht) 
                x1,y1=xlr,ylr
                #初始化 #向左下方滑动
                ori_s = ori_ss
                wt,ht =  wtt,htt                      
                xll,yll=x0,y1   
                while wt != 0 or ht != 0:
                    xt,yt = max(xll-wt,0),min(yll+ht,reso)
                    if xt == xll and yt == yll:
                        break
                    s = self.cal_score((xt,y0,x1,yt),feat[count],gt_truth[count] )
                    if s> ori_s:
                        xll,yll = xt,yt
                        wt,ht = self.cal_wt_ht((xt,y0,x1,yt))
                        ori_s = s
                    else:
                        wt,ht = int(1./2*wt),int(1./2*ht)         
                x0,y1=xll,yll
                output_list.append((x0,y0,x1,y1))
        else:
            raise Exception('adaptive_version must be 0 or 1 or 2')
        return output_list
        #print(cor_list)
        #print(output_list)

    def cal_score(self , cor , feat_map,gt_truth):
        '''
        input:
        cor: (x0,y0,x1,y1) ; feat_map: (512,28,28) ; gt_truth: (1,)tensor
        the element in cor between 0-27 you should +1 in the slice operation
        score: int 滑动得分
        '''
        x0 , y0 , x1 , y1 = cor
        feat_map = feat_map[:,y0:y1+1 , x0:x1+1]#(512,h,w)
        if 0 in feat_map.shape:
            print(x0,y0,x1,y1)
            print(feat_map.shape)
        feat_map = feat_map.unsqueeze(1)#(512,1,h,w)

        feat_map = F.interpolate(feat_map , (self.resolution,self.resolution) ,  mode = 'bilinear',align_corners=False)
        #(512,1,28,28)
        feat_map = feat_map.squeeze(dim=1).unsqueeze(0) #(1,512,28,28)
        score,_ = self.clss_process(feat_map,'a')#这里用直接取出来的值计算
        score = score[0,int(gt_truth)].item()#这里为什么要加int？？？
        return score

    def cal_wt_ht(self , cor):
        x0 , y0 , x1 , y1 = cor
        w = int((1./2)*(x1-x0))
        h = int((1./2)*(y1-y0))
        return (w,h)

    def restore(self , heatmp , cor_list , i):
        #heatmp (batch,200,28,28) ; cor_list (list style) (batch,)

        batch = heatmp.shape[0]
        output = Variable(torch.zeros((batch,1024,self.resolution,self.resolution)).cuda())
        for idx in range(batch):            
            x0,y0,x1,y1 = self.cal_idx( cor_list[idx] , i) 
            size = (y1+1-y0,x1+1-x0 )
            heatmp_idx = heatmp[idx].unsqueeze(1) #(200,1,28,28)
            heatmp_idx =  F.interpolate(heatmp_idx , size , mode = 'bilinear',align_corners=False)
            output[idx,:,y0:y1+1,x0:x1+1] = heatmp_idx.squeeze(dim=1) #(200,w,h)
        return output #(batch,200,28,28)

    def cal_idx(self ,cordinate , i):
        reso = self.resolution-1
        x0,y0,x1,y1 = cordinate
        if i == 0:
            pass
        elif i ==1 :
            y1 = min(y1+5,reso)
        elif i ==2 :
            y0 =  max(y0-5,0)
        elif i ==3 :
            x0 = max(x0-5,0)
        elif i ==4 :
            x1 = min(x1+5,reso)
        elif i ==5 :
            y0 = max(y0-5,0)
            y1 = min(y1+5,reso)
            x0 = max(x0-5,0)
            x1 = min(x1+5,reso)
        elif i ==6 :
            y0 = max(y0-10,0)
            y1 = min(y1+10,reso)
            x0 = max(x0-10,0)
            x1 = min(x1+10,reso)
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
        feat_resize = torch.squeeze(F.interpolate(cur_slide , (self.resolution,self.resolution), mode = 'bilinear',align_corners=False))
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
        slide_list.append(feature_map[:,y0:min(y1+6,self.resolution),x0:x1+1])
        slide_list.append(feature_map[:,max(y0-5,0):y1+1,x0:x1+1])
        slide_list.append(feature_map[:,y0:y1+1,max(x0-5,0):x1+1])
        slide_list.append(feature_map[:,y0:y1+1,x0:min(x1+6,self.resolution)])
        slide_list.append(feature_map[:,max(y0-5,0):min(y1+6,self.resolution),max(x0-5,0):min(x1+6,self.resolution)])
        slide_list.append(feature_map[:,max(y0-10,0):min(y1+11,self.resolution),max(x0-10,0):min(x1+11,self.resolution)])
        count = 0
        for cur_slide in slide_list:
            count +=1
            cur_slide = cur_slide.unsqueeze(1) #(512,1,w,h)
            cur_resize = torch.squeeze(F.interpolate(cur_slide , (self.resolution,self.resolution), mode = 'bilinear',align_corners=False))
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
            step1用torch将>thr部分置为0，<thr部分置为255，再np化来找区域
        '''
        atten_map = (atten_map_normed>threshold)*255
        b,width,height = atten_map_normed.shape
        #for i in range(b):
        #    self.save_attenn(atten_map[i,:,:])

        #atten_map = atten_map*255
        np_atten = atten_map.cpu().numpy()

        cor_list = []

        for i in range(b):
            contours = cv2.findContours(
                image=np_atten[i,:,:].astype('uint8'),
                mode=cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]
            #contour = [max(contours, key=cv2.contourArea)][0]
            rx0,ry0,rx1,ry1 = width,height,0,0
            for contour in contours:
                x,y,w,h = cv2.boundingRect(contour) #计算contour的边界
                x0,y0,x1,y1 = x,y,x+w,y+h
                rx1 = max(x1,rx1)
                ry1 = max(y1,ry1)
                rx0 = min(x0,rx0)
                ry0 = min(y0,ry0)
            cor_list.append([rx0,ry0,min(rx1,width-1),min(ry1,height-1)])
        return cor_list
    
    def save_attenn(self,atten_map):
        a = atten_map.tolist()
        with open('/project/zmwang/qxy/DJL/save_bins/map.txt','a') as f:
            for i in a:
                json.dump([round(x,2)for x in i],f)
                f.write('\n')
            f.write('\n')

    def find_highlight_region2(self , atten_map_normed , threshold ):
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
    def get_loss(self, logits, gt_labels,current_epoch,dataset):
        if self.onehot == 'True':
            gt = gt_labels.float()
        else:
            gt = gt_labels.long()
        loss_cls = self.loss_cross_entropy(logits[0], gt)
        loss_cls_ers = self.loss_cross_entropy(logits[1], gt)
        if sys.version_info.major == 3:
            booltensor = torch.zeros( (logits[0].shape[0],logits[0].shape[1]) ).type(torch.bool)
        elif sys.version_info.major == 2:
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
        return map_erase
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
