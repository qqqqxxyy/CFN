#-*- coding: UTF-8 -*-
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import numpy as np
import ipdb
import sys
from utils.utils import to_image, acc_counter,loss_counter
sys.path.append('../')


__all__ = [
    'get_model',
]

def plus(a,b):
    return a+b

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

    def __init__(self, features, num_classes=1000, args=None, threshold=0.6):
        super(VGG, self).__init__()
        self.features = features
        self.clas = self.classifier(num_classes)
        self.clas_fc = nn.Linear(1024,num_classes)
        #self.conv6 = nn.Conv2d(512,1024,kernel_size=3, padding=1)
        #self.relu = nn.ReLU(inplace=False)
        #self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.fc = nn.Linear(1024,num_classes)
        self.initialize_weights(self.modules(), init_mode='he')
        self.loss_cross_entropy = nn.CrossEntropyLoss()
    
    def classifier(self,num_classes):
        return nn.Sequential(
            nn.Conv2d(512,1024,kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool2d(1)
        )


    def forward(self, x, label=torch.tensor([1]) , return_cam=False):
        x = self.features(x)
        x = self.clas[0](x)
        x = self.clas[1](x)
        pre_logit = self.clas[2](x)
        #x = self.conv6(x)
        #x = self.relu(x)
        #pre_logit = self.avgpool(x)
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
            return cams,[logits,],feature_map
        return [logits, ]

    def get_loss(self, logits, gt_labels,current_epoch):
        gt = gt_labels.long()
        loss_cls = self.loss_cross_entropy(logits[0], gt)
        return [loss_cls, ]

    def get_localization_maps(self):
        cams = self.cams #only return map_erase
        return cams

    def initialize_weights(self, modules, init_mode):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                if init_mode == 'he':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                            nonlinearity='relu')
                elif init_mode == 'xavier':
                    nn.init.xavier_uniform_(m.weight.data)
                else:
                    raise ValueError('Invalid init_mode {}'.format(init_mode))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def get_heatmaps(self):
        return self.cams.detach().cpu().data.numpy().astype(np.float)

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


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight.data)
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
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]#几乎折半
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]#不改变尺寸
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
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'N'],
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
    #this is what fe
    model = VGG(make_layers(cfg['D1'], dilation=dilation['D1']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model

