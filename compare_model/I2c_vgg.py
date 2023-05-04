#-*- coding: UTF-8 -*-
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import numpy as np
import ipdb
#from models.obj_simi import obj_simi

# from extensions.functions.function import Edge_loss

import sys
sys.path.append('../')


__all__ = [
    'get_model',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, args=None, threshold=None):
        super(VGG, self).__init__()
        self.features = features

        self.cls_fc6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),   #fc6
            nn.ReLU(True),
        )
        self.cls_fc7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),   #fc6
            nn.ReLU(True))
        self.cls_fc8 = nn.Conv2d(1024, num_classes, kernel_size=1, padding=0)  #fc8
        self.th = threshold

        self.center_feat_bank = nn.Parameter(torch.randn((num_classes, 512)), requires_grad=False)
        self.counter=nn.Parameter(torch.zeros(num_classes), requires_grad=False)
        # self.aux_cls = nn.Linear(768, args.num_classes)


        self.loss_local_factor = args.loss_local_factor
        self.local_seed_num = args.local_seed_num
        self.loss_global_factor = args.loss_global_factor
        self._initialize_weights()

        self.onehot = False

        # self.edge_loss = Edge_loss()

    def inference(self, x):
        if self.training:
            x = F.dropout(x, 0.5)
        x = self.cls_fc6(x)

        if self.training:
            x = F.dropout(x, 0.5)
        x = self.cls_fc7(x)

        if self.training:
            x = F.dropout(x, 0.5)
        out1 = self.cls_fc8(x)
        return out1, x


    def forward(self, x, gt_label=torch.tensor([1]), edge_map=None):
        #backbone
        x = self.features(x)
        feat4 = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        #直接通过inference得到第一个feat
        out1, last_feat = self.inference(feat4)
        logits_1 = torch.mean(torch.mean(out1, dim=2), dim=2)
        if self.training==False:
            gt_label = torch.tensor([torch.argmax(logits_1,1).item()])
            gt_label = gt_label.long().detach().clone()
        self.atten_map = self.get_atten_map(out1, gt_label, True) # normalized attention maps
        self.map1 = out1

        if self.training:
            b, w, h = self.atten_map.size()
            mask = torch.zeros( (logits_1.size()[0], w, h) ).fill_(0).cuda()
        
            mask = self.get_mask(mask, self.atten_map) #mask标记处阈值较高的图片

            obj_inds = mask.new_zeros((b, self.local_seed_num)) #obj_inds(b,n) 
            obj_inds = self.obj_valid_inds(mask,obj_inds) #currently just insure code runing to take top3 indexes
            #obj_simi.obj_valid_inds(mask, obj_inds)
            obj_inds = obj_inds.long()

            obj_out1 = []
            b, c, w,h = feat4.size()
            for i in range(b):
                obj_out1.append(feat4[i].view(c,-1)[:,obj_inds[i]])
            obj_out1 = torch.stack(obj_out1)

            b, c, n = obj_out1.size() #b:batch_size  c:channel 512  n:number of seed 3
            try:
                tmp1 = obj_out1.view(-1,2, c, n)[:,0,:,:] 
                tmp2 = obj_out1.view(-1,2, c, n)[:,1,:,:]
            except RuntimeError:
                print(b,c,n)
                ipdb.set_trace()
            tmp1 = tmp1.permute(0,2,1).contiguous().view(-1, c) 
            tmp2 = tmp2.permute(0,2,1).contiguous().view(-1, c)
            pixel_loss = F.pairwise_distance(tmp1, tmp2, 2)

            cls_center_feat = obj_out1.view(-1, 2, c, n).mean(dim=3).mean(dim=1)
            #(6,512,3) -> (3,2,512,3) -> (3,512) 把同一类别的batch向量取了均值
            # aux_logits = self.center_aux_classifier(gt_label, cls_center_feat)

            return [logits_1, pixel_loss, cls_center_feat]
        else:
            pixel_loss = 0
            return self.atten_map,logits_1#, self.get_localization_maps()]
    
    def obj_valid_inds(self , mask,obj_inds):
        '''obj_inds : (b,k)
        '''
        batch_size = mask.size()[0] #获取batch_size信息
        num = obj_inds.size()[1] #获取随机取元素信息

        #batch 中每个元素都要单独讨论
        for i in range(batch_size):
            pixel_inds = mask[i].view(-1).nonzero();#非0元素的序号
            nonzero_num = pixel_inds.size(0); #统计非零元素总个数
            rand_indx = torch.randperm(nonzero_num)
            if rand_indx.shape[0]<=3:
                ipdb.set_trace()
            for j in range(num):
                obj_inds[i,j] = pixel_inds[rand_indx[j]] #原论文这里，如果选的vector不够需要的数目，做了取模处理。
                #而我们处理高亮点不够随机数的方法是，1.随机num很小只有3，这也是论文呢报告的较好的
        
        
        return obj_inds


    def get_localization_maps(self):
        map1 = self.normalize_atten_maps(self.map1)
        # map_erase = self.normalize_atten_maps(self.map_erase)
        # return torch.max(map1, map_erase)
        return map1

    def mark_obj(self, label_img, heatmap, label, threshold=0.5):
        '''input:
        label_img : (b,28,28) 全0
        heatmap : (b,28,28)
        output：
        label_img : (b,28,28) 前景信息标记为label的数值（默认为1）
        '''
        #这个if判断和下面的if只有一个效果，就是可以控制
        #标记的数值，不一定为0-1标记，可以认为指定其他的数字
        if isinstance(label, (float, int)):
            np_label = label
        else:
            np_label = label.cpu().data.numpy().tolist()

        for i in range(heatmap.size()[0]):
            mask_pos = heatmap[i] > threshold
            if torch.sum(mask_pos.float()).data.cpu().numpy() < 30:
                threshold = torch.max(heatmap[i]) * 0.5
                mask_pos = heatmap[i] > threshold
                if torch.sum(mask_pos.float()).data.cpu().numpy() <= 3:
                    threshold = torch.max(heatmap[i]) * 0.1
                    mask_pos = heatmap[i] > threshold
            label_i = label_img[i]
            if isinstance(label, (float, int)):
                use_label = np_label
            else:
                use_label = np_label[i]
            # label_i.masked_fill_(mask_pos.data, use_label)
            label_i[mask_pos.data] = use_label
            label_img[i] = label_i

        return label_img


    def mark_bg(self, label_img, heatmap, threshold=0.1):
        mask_pos = heatmap < threshold
        # label_img.masked_fill_(mask_pos.data, 0.0)
        label_img[mask_pos.data] = 0.0

        return label_img

    def get_mask(self, mask, atten_map, th_high=0.7, th_low = 0.05):
        #mask label for segmentation
        mask = self.mark_obj(mask, atten_map, 1.0, th_high)
        mask = self.mark_bg(mask, atten_map, th_low)

        return  mask


    def update_center_vec(self, gt_labels, center_feats):
        batch_size = gt_labels.size(0)
        unique_gt_labels = gt_labels.view(int(batch_size/2), 2)[:,0].long()

        lr = torch.exp(-0.002*self.counter[unique_gt_labels])
        # lr = torch.exp(-0.001*self.counter[unique_gt_labels])
        fused_centers = (1 - lr.detach()).unsqueeze(dim=1)*self.center_feat_bank[unique_gt_labels].detach() + lr.detach().unsqueeze(dim=1)*center_feats.detach()
        self.counter[unique_gt_labels] += 1
        self.center_feat_bank[unique_gt_labels] = fused_centers
        return fused_centers



    def get_loss(self, logits, gt_labels,current_epoch,dataset):
        cls_logits, pixel_loss, batch_center_vecs  = logits
        gt = gt_labels.long()
        loss_cls = F.cross_entropy(cls_logits, gt)#分类的损失分数
        #ipdb.set_trace()

        batch_size = gt_labels.size(0)
        unique_gt_labels = gt_labels.view(int(batch_size/2), 2)[:,0].long() #将gt_label排列为两派取第一列
        # aux_loss_cls = F.cross_entropy(aux_logits, unique_gt_labels.long())
        #从center_feat_bank取出对应类别计算loss
        aux_loss_cls = F.pairwise_distance(self.center_feat_bank[unique_gt_labels].cuda().detach(), batch_center_vecs, 2)
        #更新向量
        self.update_center_vec(gt_labels, batch_center_vecs.detach())
        if dataset=='CUB':
            if current_epoch < 10:
                loss = loss_cls
            else :
                loss = loss_cls + self.loss_global_factor*aux_loss_cls.mean()+ self.loss_local_factor*pixel_loss.mean() #
        elif dataset=='ILSVRC':
            if current_epoch == 0:
                loss = loss_cls
            else :
                loss = loss_cls + self.loss_global_factor*aux_loss_cls.mean()+ self.loss_local_factor*pixel_loss.mean() #
        else:
            raise Exception('wrong dataset input{}, it must be CUB or ILSVRC'.format(dataset))
        return loss, loss_cls, self.loss_local_factor*pixel_loss.mean()#, 

    def get_all_localization_maps(self):
        return self.normalize_atten_maps(self.map1)

    def get_heatmaps(self, gt_label, map1):
        map1 = self.get_atten_map(map1, gt_label)
        return [map1,]

    def get_fused_heatmap(self, gt_label):
        maps = self.get_heatmaps(gt_label=gt_label)
        fuse_atten = maps[0]
        return fuse_atten

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
        label = gt_labels.long()

        feature_map_size = feature_maps.size()
        batch_size = feature_map_size[0]

        atten_map = torch.zeros([feature_map_size[0], feature_map_size[2], feature_map_size[3]])
        atten_map = Variable(atten_map.cuda())
        for batch_idx in range(batch_size):
            atten_map[batch_idx,:,:] = torch.squeeze(feature_maps[batch_idx, label.data[batch_idx], :,:])

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

def make_layers(cfg, dilation=None, batch_norm=False):
    layers = []
    in_channels = 3
    for v, d in zip(cfg, dilation):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d, dilation=d)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

dilation = {
    'D1': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'N', 1, 1, 1, 'N']
}


def model(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    """
    model = VGG(make_layers(cfg['D1'], dilation=dilation['D1']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model

