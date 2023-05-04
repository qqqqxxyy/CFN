# License: https://github.com/milesial/Pytorch-UNet
""" Full assembly of the parts to form the complete network """

from torch import nn
from UNet.unet_parts import DoubleConv, Down, Up, OutConv
import ipdb
import torch
from torch.nn import CrossEntropyLoss 
from utils.utils import acc_counter,loss_counter
import torch.nn.functional as F
import numpy as np
'''
models in this package are all trained with unet added a CAM on the end of decoder.
the different of models is different interactions between CAMs branch and seg branch.
'''
class UNet_cam_base(nn.Module):
    def __init__(self, n_channels=3, out_channels=2, bilinear=True):
        super(UNet_cam_base, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)
        self.classifier = self._clas(1000)
        self.phase = None
    def forward(self, x,label=None):
        # print('x shape:',x.shape)
        x1 = self.inc(x)
        # print('x1 shape:',x1.shape)
        x2 = self.down1(x1)
        # print('x2 shape:',x2.shape)
        x3 = self.down2(x2)
        # print('x3 shape:',x3.shape)
        x4 = self.down3(x3)
        # print('x4 shape:',x4.shape)
        x5 = self.down4(x4)
        # print('x5 shape:',x5.shape)
        x = self.up1(x5, x4)
        # print('x(x5,x4) shape:',x.shape)
        x = self.up2(x, x3)
        # print('x(x,x3) shape:',x.shape)
        x = self.up3(x, x2)
        self.secfeat = x
        # print('x(x,x2) shape:',x.shape)
        x = self.up4(x, x1)
        self.firfeat = x
        # print('x(x,x1) shape:',x.shape)#对此处x进行classifier
        # ipdb.set_trace()
        self.tmp = x
        self.x_cls = self.classifier(x)

        # print('x_cls:',self.x_cls.shape)
        self.logits = self.outc(x)
        # print('logits:', self.logits.shape)
        # ipdb.set_trace()
        if self.training==False:
            feature = self.classifier[0:5](x).detach().clone()
            label = self.x_cls.argmax(1).item()
            #label = label.long().detach().clone().item()
            # print(label)
            cams_weight = self.classifier[7].weight[label]
            batchs, dims = feature.shape[:2]
            cams = (cams_weight.view(batchs,dims, 1, 1) *feature
                ).mean(1, keepdim=False)
            self.cams = cams
            return 
        # print(logits.shape)
        # ipdb.set_trace()
        return
        
    def down1_up4(self,x1):

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

    def CAM(self,x):
        # ipdb.set_trace()
        feature = self.classifier[0:5](x)
        # print(feature.shape)
        label = self.x_cls.argmax(1).detach().clone()
        # print(label.shape)
        cams_weight = self.classifier[7].weight[label]
        # print(cams_weight.shape)
        batchs, dims = feature.shape[:2]
        # print(batchs)
        cams = (cams_weight.view(batchs,dims, 1, 1) *feature
            ).mean(1, keepdim=False)
        # print(cams.shape)
        return cams

    def _clas(self, num_cls=200):
        clas = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(4),
            nn.Conv2d(64,1024,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            self._Flatten(),
            nn.Linear(in_features=1024,out_features=num_cls,bias=True)
        )
        return clas

    class _Flatten(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self,x):
            #返回后面所有元素的乘积
            shape = torch.prod(torch.tensor(x.shape[1:])).item()
            return x.reshape(-1,shape)

    def get_loss(self,cls_pre,cls_ture,htmap):
        CEL = CrossEntropyLoss()
        #loss_seg = CEL(seg_pre,seg_true)
        loss_cls = CEL(cls_pre,cls_ture.long())
        # H = htmap
        # miu = 0.5
        # sigma = 0.1
        # R = torch.exp( -(H-miu)**2/(2*sigma**2) )
        # loss_htmp = ((-H*torch.log(H)-(1-H)*torch.log(1-H))*R).mean()
        # loss = loss_cls+1*loss_htmp
        #return loss_seg+0.5*loss_cls,loss_seg,loss_cls
        return [loss_cls]

class UNet_fb_v1(UNet_cam_base):
    
    def __init__(self, n_channels=3, out_channels=2, bilinear=True):
        super(UNet_fb_v1, self).__init__(n_channels,out_channels,bilinear)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up22 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.phase = None
    def forward(self,x,label=None):
        super(UNet_fb_v1,self).forward(x,label)
        # if self.training == True:
        #     if self.phase == 1:
        #         return self.x_cls,self.logits
        #     elif self.phase == 2:
        #         mask_pre = ((1.0 - torch.softmax(self.logits, dim=1))[:, 0]>0.5).float().unsqueeze(1)
        #         x_f = (x*mask_pre).detach()
        #         x_b = (x*(1-mask_pre)).detach()
        #         x_f = x_f.contiguous()

        #         x_f1 = self.inc(x_f)
        #         x_fo = self.down1_up4(x_f1)
        #         x_f_cls = self.classifier(x_fo)

        #         x_b1 = self.inc(x_b)
        #         x_bo = self.down1_up4(x_b1)
        #         x_b_cls = self.classifier(x_bo)

        #         return self.x_cls,self.logits,x_f_cls,x_b_cls
        #     elif self.phase == 3:
        #         mask_pre = ((1.0 - torch.softmax(self.logits, dim=1))[:, 0]).unsqueeze(1)
        #         x_f = (x*mask_pre).contiguous()
        #         x_b = x*(1-mask_pre)

        #         x_f1 = self.inc(x_f)
        #         x_fo = self.down1_up4(x_f1)
        #         x_f_cls = self.classifier(x_fo)

        #         x_b1 = self.inc(x_b)
        #         x_bo = self.down1_up4(x_b1)
        #         x_b_cls = self.classifier(x_bo)

        #         return self.x_cls,self.logits,x_f_cls,x_b_cls

        # else:
        #     return self.logits,self.cams,self.x_cls


    def get_loss(self,logits,cls_true,seg_true,index):
        cls_pre,seg_pre = logits[0],logits[1]
        CEL = CrossEntropyLoss()
        half_len = cls_true.shape[0]
        cls_pre = cls_pre[:half_len,:]
        seg_pre = seg_pre[half_len:,:]
        loss_cls = CEL(cls_pre,cls_true.long())

        loss_seg = CEL(seg_pre,seg_true)
        if self.phase == 1:
            loss = 0.5*loss_cls+loss_seg
            return loss,loss_seg,0.,0.,loss_cls
        else:
            cls_fore, cls_back = logits[2],logits[3]
            cls_fore = cls_fore[:half_len,:]
            cls_back = cls_back[:half_len,:]

            loss_back = torch.softmax(cls_back,dim=1)*\
                torch.log(torch.softmax(cls_back,dim=1)+1e-6)
            loss_back = loss_back.mean()
            loss_fore = CEL(cls_fore,cls_true)
            
            loss = 0.5*loss_cls+loss_seg+loss_fore+loss_back
            return loss,loss_seg,loss_fore,loss_back,loss_cls

    def loss_init(self):
        self.loss_ct = []
        for i in range(5):
            self.loss_ct.append(loss_counter())
        self.loss_ct.append(acc_counter())

    def loss_count(self,logits,cls_pre,clas):
        for i in range(5):
            self.loss_ct[i].count(logits[i])
        self.loss_ct[5].count(cls_pre[:clas.shape[0],:].argmax(1).cpu().numpy(),clas.cpu().numpy())

    def loss_reset(self):
        for i in range(6):
            self.loss_ct[i].reset()

    def loss_output(self):
        loss = self.loss_ct[0]()
        loss_seg = self.loss_ct[1]()
        loss_fore = self.loss_ct[2]()
        loss_back = self.loss_ct[3]()
        loss_cls = self.loss_ct[4]()
        clas = self.loss_ct[5]()
        output = ('loss:{:.4f},loss_seg:{:.4f},loss_fore:{:.4f},loss_back:{:.4f},'+\
                    'loss_cls:{:.4f},cls:{:.4f}')\
            .format(loss,loss_seg,loss_fore,loss_back,loss_cls,clas)

        return output

class UNet_fb_v3(UNet_fb_v1):
    def forward(self,x,label=None):
        super(UNet_fb_v1,self).forward(x,label)
        if self.training == True:
            if self.phase == 1:
                return self.x_cls,self.logits
            elif self.phase == 2:
                mask_pre = (1.0 - torch.softmax(self.logits, dim=1))[:, 0].unsqueeze(1)
                x_f = x*mask_pre
                # x_b = (x*(1-mask_pre)).detach()
                # x_f = (x_f.contiguous()).detach()
                x_b = (x*(1-mask_pre))
                x_f = (x_f.contiguous())
                x_f1 = self.inc(x_f)
                x_fo = self.down1_up4(x_f1)
                x_f_cls = self.classifier(x_fo)

                x_b1 = self.inc(x_b)
                x_bo = self.down1_up4(x_b1)
                x_b_cls = self.classifier(x_bo)
                
                return self.x_cls,self.logits,x_f_cls,x_b_cls
            else:
                featmp = self.tmp
                mask_pre = (1.0 - torch.softmax(self.logits, dim=1))[:, 0].unsqueeze(1)
                # mask_pre = torch.max_pool2d(mask_pre,2).unsqueeze(1)
                # ipdb.set_trace()
                ftmp_fo = featmp*mask_pre
                ftmp_bc = featmp*(1.0-mask_pre)

                x_f_cls = self.classifier(ftmp_fo)
                x_b_cls = self.classifier(ftmp_bc)
                return self.x_cls,self.logits,x_f_cls,x_b_cls
        else:
            mask_pre = (1.0 - torch.softmax(self.logits, dim=1))[:, 0]
            return mask_pre, self.x_cls           

    def initialize(self,marker):
        '''make branch A detach or attach of training
        '''
        layer_lis = [self.classifier]
        for p in layer_lis:
            for q in p.parameters():
                if marker == 'det':
                    q.requires_grad=False
                if marker == 'att':
                    q.requires_grad=True

    def get_loss(self,logits,cls_true,seg_true):
        'logits stands for the all outputs from forward propogation'
        cls_pre,seg_pre = logits[0],logits[1]
        CEL = CrossEntropyLoss()
        half_len = cls_true.shape[0]
        cls_pre = cls_pre[:half_len,:]
        seg_pre = seg_pre[half_len:,:]
        # loss_cls = CEL(cls_pre,cls_true.long())

        loss_seg = CEL(seg_pre,seg_true)
        if self.phase == 1:
            loss = loss_seg

            return loss,loss_seg,0.,0.,0.
        else:

            cls_fore, cls_back = logits[2],logits[3]
            cls_fore = cls_fore[:half_len,:]
            cls_back = cls_back[:half_len,:]
            loss_back = torch.softmax(cls_back,dim=1)*\
                torch.log(torch.softmax(cls_back,dim=1)+1e-6)
            loss_back = loss_back.mean()
            loss_fore = CEL(cls_fore,cls_true)
            loss = loss_seg+loss_fore+loss_back
            return loss,loss_seg,loss_fore,loss_back,0.

class UNet_fb_v3_1(UNet_fb_v3):
    def get_loss(self,logits,cls_true,seg_true):
        'logits stands for the all outputs from forward propogation'
        cls_pre,seg_pre = logits[0],logits[1]
        CEL = CrossEntropyLoss()
        half_len = cls_true.shape[0]
        cls_pre = cls_pre[:half_len,:]
        seg_pre = seg_pre[half_len:,:]
        # loss_cls = CEL(cls_pre,cls_true.long())
        mask_pre = (1.0 - torch.softmax(self.logits, dim=1))
        gibbs_entropy = (-mask_pre*torch.log(mask_pre+1e-8)).mean()
        loss_seg = CEL(seg_pre,seg_true)
        if self.phase == 1:
            loss = loss_seg+gibbs_entropy

            return loss,loss_seg,gibbs_entropy,0.,0.
        else:

            cls_fore, cls_back = logits[2],logits[3]
            cls_fore = cls_fore[:half_len,:]
            cls_back = cls_back[:half_len,:]
            loss_back = torch.softmax(cls_back,dim=1)*\
                torch.log(torch.softmax(cls_back,dim=1)+1e-6)
            loss_back = loss_back.mean()
            loss_fore = CEL(cls_fore,cls_true)
            loss = loss_seg+loss_fore+loss_back+gibbs_entropy
            return loss,loss_seg,loss_fore,loss_back,gibbs_entropy

class UNet_fb_v3_2(UNet_fb_v3):
    def get_loss(self,logits,cls_true,seg_true):
        'logits stands for the all outputs from forward propogation'
        cls_pre,seg_pre = logits[0],logits[1]
        CEL = CrossEntropyLoss()
        half_len = cls_true.shape[0]
        cls_pre = cls_pre[:half_len,:]
        seg_pre = seg_pre[half_len:,:]
        # loss_cls = CEL(cls_pre,cls_true.long())
        mask_pre = (1.0 - torch.softmax(self.logits, dim=1))
        gibbs_entropy = (-mask_pre*torch.log(mask_pre+1e-8)).mean()
        loss_seg = CEL(seg_pre,seg_true)
        lac = mask_pre[:, 0].mean()
        if self.phase == 1:
            loss = loss_seg+gibbs_entropy+lac

            return loss,loss_seg,gibbs_entropy,lac,0.
        else:

            cls_fore, cls_back = logits[2],logits[3]
            cls_fore = cls_fore[:half_len,:]
            cls_back = cls_back[:half_len,:]
            loss_back = torch.softmax(cls_back,dim=1)*\
                torch.log(torch.softmax(cls_back,dim=1)+1e-6)
            loss_back = loss_back.mean()
            loss_fore = CEL(cls_fore,cls_true)
            loss = loss_seg+loss_fore+loss_back+gibbs_entropy+lac
            return loss,loss_seg,loss_fore,loss_back,lac

class UNet_seg(UNet_fb_v1):
    def forward(self,x,label=None):
        x = self.inc(x)
        x = self.down1_up4(x)
        logits = self.outc(x)
        if self.training == False:
            mask_pre = (1.0 - torch.softmax(logits, dim=1))[:, 0]
            return mask_pre,0
        return logits

    def get_loss(self,logits,seg):
        CEL = CrossEntropyLoss()
        loss = CEL(logits,seg)
        return [loss]

    def loss_init(self):
        self.loss_ct = [loss_counter()]

    def loss_count(self,logits):
        self.loss_ct[0].count(logits)

    def loss_reset(self):
        self.loss_ct[0].reset()

    def loss_output(self):
        loss = self.loss_ct[0]()
        output = ('loss:{:.4f}')\
            .format(loss)
        return output