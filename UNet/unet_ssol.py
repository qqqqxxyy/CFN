# License: https://github.com/milesial/Pytorch-UNet
""" Full assembly of the parts to form the complete network """

from os import supports_follow_symlinks
from torch import nn
from UNet.unet_parts import DoubleConv, Down, Up, OutConv
import ipdb
import torch
from torch.nn import CrossEntropyLoss 
from utils.utils import acc_counter,loss_counter
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
import random
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
        self.classifier = self._clas(2)
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

class UNet_cls_base(nn.Module):
    '''Model consists only one classification head
    For ilsvrc pretraining
    '''
    def __init__(self, n_channels=3, out_channels=2, bilinear=True):
        super(UNet_cls_base, self).__init__()
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

    def initialize(self,marker):
        pass

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
        return self.x_cls
        
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

    def get_loss(self,cls_pre,cls_ture):
        CEL = CrossEntropyLoss()
        #loss_seg = CEL(seg_pre,seg_true)
        loss_cls = CEL(cls_pre,cls_ture.long())
        return [loss_cls]

    def loss_init(self):
        self.loss_ct = []
        self.loss_ct.append(loss_counter())
        self.loss_ct.append(acc_counter())

    def loss_count(self,logits,cls_pre,clas):

        self.loss_ct[0].count(logits[0])
        self.loss_ct[1].count(cls_pre[:clas.shape[0],:].argmax(1).cpu().numpy(),clas.cpu().numpy())

    def loss_reset(self):
        for i in range(2):
            self.loss_ct[i].reset()

    def loss_output(self):
        loss = self.loss_ct[0]()
        clas = self.loss_ct[1]()
        output = ('loss:{:.4f},cls:{:.4f}')\
            .format(loss,clas)
        return output

class UNet_fb_v1(UNet_cam_base):
    
    def __init__(self, n_channels=3, out_channels=2, bilinear=True):
        super(UNet_fb_v1, self).__init__(n_channels,out_channels,bilinear)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up22 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.phase = None
    def forward(self,x,label=None):
        super(UNet_fb_v1,self).forward(x,label)



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
        if clas.shape[0]==0:
            '''if not real image processed, cls acc set as 0
            '''
            self.loss_ct[5].length = 1
        else:
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

class UNet_ssol_v1(UNet_fb_v1):
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
                ftmp_fo = featmp*mask_pre
                ftmp_bc = featmp*(1.0-mask_pre)

                x_f_cls = self.classifier(ftmp_fo)
                x_b_cls = self.classifier(ftmp_bc)
                return self.x_cls,self.logits,x_f_cls,x_b_cls
        else:
            mask_pre = (1.0 - torch.softmax(self.logits, dim=1))[:, 0]
            return mask_pre, self.x_cls           

    def get_loss(self,logits,cls_true,seg_true,index):
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
            return loss,loss_seg, 0. ,0. ,0.
        else:
            cls_fore, cls_back = logits[2],logits[3]
            cls_fore = cls_fore[:half_len,:]
            cls_back = cls_back[:half_len,:]

            cls_T_fore = torch.ones(cls_fore.shape[0]).long().cuda()
            cls_T_back = torch.zeros(cls_back.shape[0]).long().cuda()
            loss_fore = CEL(cls_fore,cls_T_fore)
            loss_back = CEL(cls_back,cls_T_back)
            loss = 2*loss_seg+loss_fore+loss_back
            return loss,loss_seg,loss_fore,loss_back,0.

class UNet_ssol_base1(nn.Module):
    def __init__(self, n_channels=3, out_channels=2, bilinear=True):
        super(UNet_ssol_base1, self).__init__()
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
        self.A_outc = OutConv(64, out_channels)
        self.B_outc = OutConv(64, out_channels)

    def backbone(self,x1):

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return x

    def forward(self,x):
        x = self.inc(x)
        featmap = self.backbone(x)
        mask_A = self.A_outc(featmap)
        mask_B = self.B_outc(featmap)
        if self.training == False:
            if self.phase == 1:
                #phase 1 use head A predict bbox
                mask_pre = (1.0 - torch.softmax(mask_A, dim=1))[:, 0]
            #phase 2 use head B predict bbox
            elif self.phase == 2:
                mask_pre = (1.0 - torch.softmax(mask_B, dim=1))[:, 0]
            return mask_pre, mask_A
        return mask_A, mask_B 


    def get_loss(self,logits,half_len,True_syn_msk):
        mask_A,mask_B = logits
        CEL = CrossEntropyLoss()
        CELP = CrossEntropyLoss(reduction='none')
        syn_A = mask_A[half_len:,:]
        real_B = mask_B[:half_len,:]
        loss_syn = CEL(syn_A,True_syn_msk)
        if self.phase == 1:
            #针对生成图片进行训练
            loss = loss_syn
            return loss,loss_syn,0.
        elif self.phase == 2:
            #训练生成和真实图片
            real_A = mask_A[:half_len,:]
            
            real_msk,weight_tensor = self._soft_supervision(real_A,0.7,0.3,0.25)
            loss_real = (CELP(real_B,real_msk.squeeze().long())*weight_tensor.squeeze()).mean()
            loss = loss_syn+loss_real
            return loss,loss_syn,loss_real



    def _soft_supervision(self,tensor,high,low,weight):
        mask_pre = (1.0 - torch.softmax(tensor, dim=1))[:, 0].unsqueeze(1)
        direct = (mask_pre>high)+(mask_pre<low)
        mask_real = mask_pre>0.50
        weight_tensor = direct+(~direct*weight)
        return mask_real.detach(), weight_tensor.detach()

    def _soft_supervision_2(self,tensor,high,low,weight):
        mask_pre = (1.0 - torch.softmax(tensor, dim=1))[:, 0].unsqueeze(1)
        direct = (mask_pre>high)+(mask_pre<low)
        mask_real = mask_pre>0.35
        weight_tensor = direct+(~direct*weight)
        return mask_real.detach(), weight_tensor.detach()

    def loss_init(self):
        self.loss_ct = []
        for i in range(3):
            self.loss_ct.append(loss_counter())

    def loss_count(self,logits):
        for i in range(3):
            self.loss_ct[i].count(logits[i])

    def loss_reset(self):
        for i in range(3):
            self.loss_ct[i].reset()

    def loss_output(self):
        loss = self.loss_ct[0]()
        loss_syn = self.loss_ct[1]()
        loss_real = self.loss_ct[2]()

        output = ('loss:{:.4f},loss_syn:{:.4f},loss_real:{:.4f}')\
            .format(loss,loss_syn,loss_real)

        return output

class UNet_ssol_base2(UNet_ssol_base1):
    def __init__(self, n_channels=3, out_channels=2, bilinear=True):
        super(UNet_ssol_base2, self).__init__()
        self.inc_B = DoubleConv(n_channels, 64)
        self.down1_B = Down(64, 128)
        self.down2_B = Down(128, 256)
        self.down3_B = Down(256, 512)
        self.down4_B = Down(512, 512)
        self.up1_B = Up(1024, 256, bilinear)
        self.up2_B = Up(512, 128, bilinear)
        self.up3_B = Up(256, 64, bilinear)
        self.up4_B = Up(128, 64, bilinear)

    def backbone_B(self, x1):
        x1 = self.inc_B(x1)
        x2 = self.down1_B(x1)
        x3 = self.down2_B(x2)
        x4 = self.down3_B(x3)
        x5 = self.down4_B(x4)
        x = self.up1_B(x5, x4)
        x = self.up2_B(x, x3)
        x = self.up3_B(x, x2)
        x = self.up4_B(x, x1)
        return x

    def forward(self,x0,label=None):
        x = self.inc(x0)
        featmap = self.backbone(x)
        mask_A = self.A_outc(featmap)
        if self.phase == 1:
            if self.training == False:
                mask_pre = (1.0 - torch.softmax(mask_A, dim=1))[:, 0]
                return mask_pre,mask_A
            return mask_A,torch.tensor([])
        if self.phase == 2:
            featmap_B = self.backbone_B(x0)
            mask_B = self.B_outc(featmap_B)
            if self.training == False:
                mask_pre = (1.0 - torch.softmax(mask_B, dim=1))[:, 0]
                return mask_pre,mask_B
            return mask_A,mask_B 

    def initialize_weights(self,mark='B'):
        '''mark = A or B
        '''
        if mark == 'A':
            layer_lis = [self.inc,self.down1,self.down2,self.down3,self.down4,
            self.up1,self.up2,self.up3,self.up4,self.A_outc]
        elif mark == 'B':
            layer_lis = [self.inc_B,self.down1_B,self.down2_B,self.down3_B,self.down4_B,
            self.up1_B,self.up2_B,self.up3_B,self.up4_B,self.B_outc]
            
        for n in layer_lis:
            for m in n.modules():
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

    def initialize(self,marker):
        '''make branch A detach or attach of training
        '''
        layer_lis = [self.inc,self.down1,self.down2,self.down3,self.down4,
            self.up1,self.up2,self.up3,self.up4,self.A_outc]
        for p in layer_lis:
            try:
            #for those
                p.BatchNorm(marker)
            except:
                pass

            for q in p.parameters():
                if marker == 'det':
                    q.requires_grad=False 
                if marker == 'att':
                    q.requires_grad=True

    def get_loss(self,logits,half_len,True_syn_msk):
        mask_A,mask_B = logits
        CEL = CrossEntropyLoss()
        CELP = CrossEntropyLoss(reduction='none')
        if self.phase == 1:
            syn_A = mask_A[half_len:,:]
            loss_syn = CEL(syn_A,True_syn_msk)
            return loss_syn,loss_syn,0.
        if self.phase == 2:
            real_A = mask_A[:half_len,:]
            real_B = mask_B[:half_len,:]
            real_msk,weight_tensor = self._soft_supervision(real_A,0.7,0.3,0.25)
            loss_real = (CELP(real_B,real_msk.squeeze().long())*weight_tensor.squeeze()).mean()
            return loss_real, 0., loss_real

class UNet_ssol_base2_check(UNet_ssol_base2):
    def forward(self,x0,label=None):
        x = self.inc(x0)

        if self.phase == 1:
            if self.training == False:
                featmap = self.backbone(x)

                mask_A = self.A_outc(featmap)
                mask_pre = (1.0 - torch.softmax(mask_A, dim=1))[:, 0]
                return mask_pre,mask_A

        if self.phase == 2:
            featmap_B = self.backbone_B(x0)
            mask_B = self.B_outc(featmap_B)
            if self.training == False:
            #phase 2 use head B predict bbox
                mask_pre = (1.0 - torch.softmax(mask_B, dim=1))[:, 0]
                return mask_pre,mask_B

class UNet_ssol_base2_2(UNet_ssol_base2):
    def get_loss(self,logits,half_len,True_syn_msk):
        mask_A,mask_B = logits
        CEL = CrossEntropyLoss()
        CELP = CrossEntropyLoss(reduction='none')
        if self.phase == 1:
            syn_A = mask_A[half_len:,:]
            loss_syn = CEL(syn_A,True_syn_msk)
            return loss_syn,loss_syn,0.
        if self.phase == 2:
            real_A = mask_A[:half_len,:]
            real_B = mask_B[:half_len,:]
            real_msk,weight_tensor = self._soft_supervision(real_A,0.6,0.4,0.5)
            loss_real = (CELP(real_B,real_msk.squeeze().long())*weight_tensor.squeeze()).mean()
            return loss_real, 0., loss_real

class UNet_ssol_base2_3(UNet_ssol_base2):
    def get_loss(self,logits,half_len,True_syn_msk):
        mask_A,mask_B = logits
        CEL = CrossEntropyLoss()
        CELP = CrossEntropyLoss(reduction='none')
        if self.phase == 1:
            syn_A = mask_A[half_len:,:]
            loss_syn = CEL(syn_A,True_syn_msk)
            return loss_syn,loss_syn,0.
        if self.phase == 2:
            real_A = mask_A[:half_len,:]
            real_B = mask_B[:half_len,:]
            real_msk,weight_tensor = self._soft_supervision(real_A,0.7,0.3,0.25)
            loss_real = (CELP(real_B,real_msk.squeeze().long()).squeeze()).mean()
            return loss_real, 0., loss_real

class UNet_ssol_base2_3_1(UNet_ssol_base2):
    '''0.35
    '''
    def get_loss(self,logits,half_len,True_syn_msk):
        mask_A,mask_B = logits
        CEL = CrossEntropyLoss()
        CELP = CrossEntropyLoss(reduction='none')
        if self.phase == 1:
            syn_A = mask_A[half_len:,:]
            loss_syn = CEL(syn_A,True_syn_msk)
            return loss_syn,loss_syn,0.
        if self.phase == 2:
            real_A = mask_A[:half_len,:]
            real_B = mask_B[:half_len,:]
            real_msk,weight_tensor = self._soft_supervision_2(real_A,0.7,0.3,0.25)
            loss_real = (CELP(real_B,real_msk.squeeze().long()).squeeze()).mean()
            return loss_real, 0., loss_real

class UNet_ssol_base2_3_2(UNet_ssol_base2):
    '''+are loss
    '''
    def get_loss(self,logits,half_len,True_syn_msk):
        mask_A,mask_B = logits
        CEL = CrossEntropyLoss()
        CELP = CrossEntropyLoss(reduction='none')
        if self.phase == 1:
            syn_A = mask_A[half_len:,:]
            loss_syn = CEL(syn_A,True_syn_msk)
            return loss_syn,loss_syn,0.
        if self.phase == 2:
            real_A = mask_A[:half_len,:]
            real_B = mask_B[:half_len,:]
            real_msk,weight_tensor = self._soft_supervision(real_A,0.7,0.3,0.25)
            loss_real = (CELP(real_B,real_msk.squeeze().long()).squeeze()).mean()
            area_l = real_B*(real_B>0.5).sum()/(128*128)
            loss = loss_real+area_l
            return loss_real, 0., loss_real

class UNet_ssol_aug(UNet_ssol_base2):
    '''adding noise resistence for augumentation
    '''
    def get_loss(self,logits,logits_t,half_len,True_syn_msk):
        mask_A,mask_B = logits
        mask_A_t,mask_B_t = logits_t
        CEL = CrossEntropyLoss()
        CELP = CrossEntropyLoss(reduction='none')
        if self.phase == 1:
            syn_A = mask_A[half_len:,:]
            syn_A_t = mask_A_t[half_len:,:]
            loss_geo = ((syn_A-syn_A_t)**2).mean()
            loss_syn = CEL(syn_A,True_syn_msk)
            loss = loss_syn#+loss_geo
            return loss,loss_syn,0.#,loss_geo
        if self.phase == 2:
            real_A = mask_A[:half_len,:]
            real_B = mask_B[:half_len,:]
            real_msk,weight_tensor = self._soft_supervision(real_A,0.7,0.3,0.25)
            loss_real = (CELP(real_B,real_msk.squeeze().long()).squeeze()).mean()
            return loss_real, 0., loss_real

class UNet_ssol_aug_1(UNet_ssol_base2):
    def get_loss(self,logits,logits_t,half_len,True_syn_msk):
        mask_A,mask_B = logits
        mask_A_t,mask_B_t = logits_t
        CEL = CrossEntropyLoss()
        CELP = CrossEntropyLoss(reduction='none')
        if self.phase == 1:
            syn_A = mask_A[half_len:,:]
            syn_A_t = mask_A_t[half_len:,:]
            loss_geo = ((syn_A-syn_A_t)**2).mean()
            loss_syn = CEL(syn_A,True_syn_msk)
            loss = loss_syn+loss_geo
            return loss,loss_syn,loss_geo
        if self.phase == 2:
            real_A = mask_A[:half_len,:]
            real_B = mask_B[:half_len,:]
            real_A_t = mask_A_t[:half_len,:]
            real_B_t = mask_B_t[:half_len,:]
            loss_geo = ((real_A-real_A_t)**2).mean()
            loss_geo_b = ((real_B-real_B_t)**2).mean()
            real_msk,weight_tensor = self._soft_supervision(real_A,0.7,0.3,0.25)
            loss_real = (CELP(real_B,real_msk.squeeze().long()).squeeze()).mean()
            loss = loss_real+loss_geo
            return loss,loss_geo, loss_real

class UNet_ssol_aug_1_2(UNet_ssol_base2):
    def get_loss(self,logits,logits_t,half_len,True_syn_msk):
        mask_A,mask_B = logits
        mask_A_t,mask_B_t = logits_t
        CEL = CrossEntropyLoss()
        CELP = CrossEntropyLoss(reduction='none')
        mask_pre = (1.0 - torch.softmax(mask_A, dim=1))
        gibbs_entropy = (-mask_pre*torch.log(mask_pre+1e-8)).mean()
        lac = mask_pre[:, 0].mean()
        if self.phase == 1:
            syn_A = mask_A[half_len:,:]
            syn_A_t = mask_A_t[half_len:,:]
            loss_geo = ((syn_A[:syn_A_t.shape[0]]-syn_A_t)**2).mean()
            loss_syn = CEL(syn_A,True_syn_msk)
            loss = loss_syn+loss_geo+gibbs_entropy-lac
            return loss,loss_syn,loss_geo

            
        if self.phase == 2:
            real_A = mask_A[:half_len,:]
            real_B = mask_B[:half_len,:]
            real_A_t = mask_A_t[:half_len,:]
            real_B_t = mask_B_t[:half_len,:]
            real_msk,weight_tensor = self._soft_supervision(real_A,0.7,0.3,0.25)
            real_msk_t,_ = self._soft_supervision(real_A_t,0.7,0.3,0.25)
            IoU = self.cal_iou_clap(real_msk,real_msk_t)
            loss_real = (CELP(real_B,real_msk.squeeze().long()).squeeze())
            loss_real = (IoU*loss_real.mean((1,2))).mean()
            loss = loss_real
            return loss, loss_real,IoU.mean()

    def cal_iou(self,map_a,map_b):
        '''map_a and map_b both:
        [N,1,128,128]
        '''
        map_a, map_b = map_a.to(torch.bool), map_b.to(torch.bool)
        intersection = torch.sum(map_a * (map_a == map_b), dim=[-1, -2]).squeeze()
        union = torch.sum(map_a + map_b, dim=[-1, -2]).squeeze()
        return (intersection.to(torch.float) / union)

    def cal_iou_clap(self,map_a,map_b):
        '''map_a and map_b both:
        [N,1,128,128]
        '''
        map_a, map_b = map_a.to(torch.bool), map_b.to(torch.bool)
        intersection = torch.sum(map_a * (map_a == map_b), dim=[-1, -2]).squeeze()
        union = torch.sum(map_a + map_b, dim=[-1, -2]).squeeze()
        IoU = (intersection.to(torch.float) / union)
        IoU[IoU>0.5]=1
        return IoU

    def geo_rot_initial(self):
        hor_flp = transforms.RandomHorizontalFlip(p=1)
        ver_flp = transforms.RandomVerticalFlip(p=1)
        rot_90 = transforms.RandomRotation(degrees=(90, 90), expand=False)
        rot_180 = transforms.RandomRotation(degrees=(180, 180), expand=False)
        rot_270 = transforms.RandomRotation(degrees=(270, 270), expand=False)
        self.geo_forward = [hor_flp,ver_flp,rot_90,rot_180,rot_270,]
        self.geo_backward = [hor_flp,ver_flp,rot_270,rot_180,rot_90]

    def geo_rot_special(self):
        self.batch_size = 4
        self.random_geo = [4,4,4,4]

    def image_rotation(self,image):
        image_T = torch.zeros_like(image)
        for i in range(self.batch_size):#对每一个图片都施加随机集合变换
            image_T[i,:] = self.geo_forward[self.random_geo[i]](image[i,:])
        return image_T

    def image_revise_rotation(self,mask_A,mask_B):
        #对旋转图片施加反转
        mask_A_T = torch.zeros_like(mask_A)
        mask_B_T = torch.zeros_like(mask_B)
        for i in range(self.batch_size):
            mask_A_T[i,:] = self.geo_backward[self.random_geo[i]](mask_A[i,:])
            if mask_B_T.shape[0] != 0:
                mask_B_T[i,:] = self.geo_backward[self.random_geo[i]](mask_B[i,:])
        return mask_A_T,mask_B_T

class UNet_ssol_aug_thr07(UNet_ssol_aug_1_2):
    def cal_iou_clap(self,map_a,map_b):
        '''map_a and map_b both:
        [N,1,128,128]
        '''
        map_a, map_b = map_a.to(torch.bool), map_b.to(torch.bool)
        intersection = torch.sum(map_a * (map_a == map_b), dim=[-1, -2]).squeeze()
        union = torch.sum(map_a + map_b, dim=[-1, -2]).squeeze()
        IoU = (intersection.to(torch.float) / union)
        IoU[IoU>0.7]=1
        return IoU

class UNet_ssol_aug_thr03(UNet_ssol_aug_1_2):
    def cal_iou_clap(self,map_a,map_b):
        '''map_a and map_b both:
        [N,1,128,128]
        '''
        map_a, map_b = map_a.to(torch.bool), map_b.to(torch.bool)
        intersection = torch.sum(map_a * (map_a == map_b), dim=[-1, -2]).squeeze()
        union = torch.sum(map_a + map_b, dim=[-1, -2]).squeeze()
        IoU = (intersection.to(torch.float) / union)
        IoU[IoU>0.3]=1
        return IoU

class UNet_ssol_aug_thr03f(UNet_ssol_aug_1_2):
    def cal_iou_clap(self,map_a,map_b):
        '''map_a and map_b both:
        [N,1,128,128]
        '''
        map_a, map_b = map_a.to(torch.bool), map_b.to(torch.bool)
        intersection = torch.sum(map_a * (map_a == map_b), dim=[-1, -2]).squeeze()
        union = torch.sum(map_a + map_b, dim=[-1, -2]).squeeze()
        IoU = (intersection.to(torch.float) / union)
        IoU[IoU>0.5]=1
        IoU[IoU<0.3]=0
        return IoU

class UNet_ssol_aug_abl_05(UNet_ssol_aug_1_2):
    def get_loss(self,logits,logits_t,half_len,True_syn_msk):
        mask_A,mask_B = logits
        mask_A_t,mask_B_t = logits_t
        CEL = CrossEntropyLoss()
        CELP = CrossEntropyLoss(reduction='none')
        mask_pre = (1.0 - torch.softmax(mask_A, dim=1))
        gibbs_entropy = (-mask_pre*torch.log(mask_pre+1e-8)).mean()
        lac = mask_pre[:, 0].mean()
        if self.phase == 1:
            syn_A = mask_A[half_len:,:]
            syn_A_t = mask_A_t[half_len:,:]
            loss_geo = ((syn_A[:syn_A_t.shape[0]]-syn_A_t)**2).mean()
            loss_syn = CEL(syn_A,True_syn_msk)
            loss = loss_syn+0.5*loss_geo+gibbs_entropy-lac
            return loss,loss_syn,loss_geo

class UNet_ssol_aug_abl_08(UNet_ssol_aug_1_2):
    def get_loss(self,logits,logits_t,half_len,True_syn_msk):
        mask_A,mask_B = logits
        mask_A_t,mask_B_t = logits_t
        CEL = CrossEntropyLoss()
        CELP = CrossEntropyLoss(reduction='none')
        mask_pre = (1.0 - torch.softmax(mask_A, dim=1))
        gibbs_entropy = (-mask_pre*torch.log(mask_pre+1e-8)).mean()
        lac = mask_pre[:, 0].mean()
        if self.phase == 1:
            syn_A = mask_A[half_len:,:]
            syn_A_t = mask_A_t[half_len:,:]
            loss_geo = ((syn_A[:syn_A_t.shape[0]]-syn_A_t)**2).mean()
            loss_syn = CEL(syn_A,True_syn_msk)
            loss = loss_syn+0.8*loss_geo+gibbs_entropy-lac
            return loss,loss_syn,loss_geo

class UNet_ssol_aug_abl_15(UNet_ssol_aug_1_2):
    def get_loss(self,logits,logits_t,half_len,True_syn_msk):
        mask_A,mask_B = logits
        mask_A_t,mask_B_t = logits_t
        CEL = CrossEntropyLoss()
        CELP = CrossEntropyLoss(reduction='none')
        mask_pre = (1.0 - torch.softmax(mask_A, dim=1))
        gibbs_entropy = (-mask_pre*torch.log(mask_pre+1e-8)).mean()
        lac = mask_pre[:, 0].mean()
        if self.phase == 1:
            syn_A = mask_A[half_len:,:]
            syn_A_t = mask_A_t[half_len:,:]
            loss_geo = ((syn_A[:syn_A_t.shape[0]]-syn_A_t)**2).mean()
            loss_syn = CEL(syn_A,True_syn_msk)
            loss = loss_syn+1.5*loss_geo+gibbs_entropy-lac
            return loss,loss_syn,loss_geo

class UNet_ssol_aug_abl_20(UNet_ssol_aug_1_2):
    def get_loss(self,logits,logits_t,half_len,True_syn_msk):
        mask_A,mask_B = logits
        mask_A_t,mask_B_t = logits_t
        CEL = CrossEntropyLoss()
        CELP = CrossEntropyLoss(reduction='none')
        mask_pre = (1.0 - torch.softmax(mask_A, dim=1))
        gibbs_entropy = (-mask_pre*torch.log(mask_pre+1e-8)).mean()
        lac = mask_pre[:, 0].mean()
        if self.phase == 1:
            syn_A = mask_A[half_len:,:]
            syn_A_t = mask_A_t[half_len:,:]
            loss_geo = ((syn_A[:syn_A_t.shape[0]]-syn_A_t)**2).mean()
            loss_syn = CEL(syn_A,True_syn_msk)
            loss = loss_syn+2.0*loss_geo+gibbs_entropy-lac
            return loss,loss_syn,loss_geo


class UNet_ssol_aug_1_3(UNet_ssol_aug_1_2):
    def geo_rot_special(self):
        self.batch_size = 4
        self.random_geo = [3,3,3,3]

class UNet_ssol_aug_1_4(UNet_ssol_aug_1_2):
    def geo_rot_special(self):
        self.batch_size = 4
        self.random_geo = torch.randint(0,5,(self.batch_size,))

class UNet_ssol_aug_1_5(UNet_ssol_aug_1_2):
    def geo_rot_special(self):
        self.batch_size = 4
        self.random_geo = [4,4,4,4]

    def image_rotation(self,image):
        image_T = torch.zeros_like(image)[:self.batch_size]
        for i in range(self.batch_size):#对每一个图片都施加随机集合变换
            image_T[i,:] = self.geo_forward[self.random_geo[i]](image[i,:])
        return image_T

    def image_revise_rotation(self,mask_A,mask_B):
        mask_A_T = torch.zeros_like(mask_A)
        mask_B_T = torch.zeros_like(mask_B)
        for i in range(self.batch_size):
            mask_A_T[i,:] = self.geo_backward[self.random_geo[i]](mask_A[i,:])
            if mask_B_T.shape[0] != 0:
                mask_B_T[i,:] = self.geo_backward[self.random_geo[i]](mask_B[i,:])
        return mask_A_T,mask_B_T

class UNet_ssol_aug_1_6(UNet_ssol_aug_1_2):
    def geo_rot_special(self):
        self.batch_size = 0
        self.random_geo = []

class UNet_ssol_aug_1_7(UNet_ssol_aug_1_2):
    def get_loss(self,logits,logits_t,half_len,True_syn_msk):
        mask_A,mask_B = logits
        mask_A_t,mask_B_t = logits_t
        CEL = CrossEntropyLoss()
        CELP = CrossEntropyLoss(reduction='none')
        mask_pre = (1.0 - torch.softmax(mask_A, dim=1))
        lac = mask_pre[:, 0].mean()
        if self.phase == 1:
            syn_A = mask_A[half_len:,:]
            syn_A_t = mask_A_t[half_len:,:]
            loss_geo = ((syn_A[:syn_A_t.shape[0]]-syn_A_t)**2).mean()
            loss_syn = CEL(syn_A,True_syn_msk)
            loss = loss_syn+loss_geo
            return loss,loss_syn,loss_geo
        if self.phase == 2:
            real_A = mask_A[:half_len,:]
            real_B = mask_B[:half_len,:]
            real_A_t = mask_A_t[:half_len,:]
            real_B_t = mask_B_t[:half_len,:]
            
            real_msk,weight_tensor = self._soft_supervision(real_A,0.7,0.3,0.25)
            real_msk_t,_ = self._soft_supervision(real_A_t,0.7,0.3,0.25)
            IoU = self.cal_iou_clap(real_msk,real_msk_t)
            loss_real = (CELP(real_B,real_msk.squeeze().long()).squeeze())
            loss_real = (IoU*loss_real.mean((1,2))).mean()
            loss = loss_real
            return loss, loss_real,IoU.mean()

    def geo_rot_special(self):
        self.batch_size = 6
        self.random_geo = torch.randint(0,5,(self.batch_size,))


class UNet_ssol_aug_2(UNet_ssol_aug_1_2):
    def get_loss(self,logits,logits_t,half_len,True_syn_msk):
        mask_A,mask_B = logits
        mask_A_t,mask_B_t = logits_t
        CEL = CrossEntropyLoss()
        CELP = CrossEntropyLoss(reduction='none')
        mask_pre = (1.0 - torch.softmax(mask_A, dim=1))
        gibbs_entropy = (-mask_pre*torch.log(mask_pre+1e-8)).mean()
        lac = mask_pre[:, 0].mean()
        if self.phase == 1:
            syn_A = mask_A[half_len:,:]
            syn_A_t = mask_A_t[half_len:,:]
            loss_geo = ((syn_A[:syn_A_t.shape[0]]-syn_A_t)**2).mean()
            loss_syn = CELP(syn_A,True_syn_msk)
            drop_mask = self.random_drop(4,loss_syn)
            loss_syn = (loss_syn*drop_mask).mean()
            loss = 2*loss_syn+loss_geo+gibbs_entropy-lac
            return loss,loss_syn,loss_geo
        if self.phase == 2:
            real_A = mask_A[:half_len,:]
            real_B = mask_B[:half_len,:]
            real_A_t = mask_A_t[:half_len,:]
            real_B_t = mask_B_t[:half_len,:]
            
            real_msk,weight_tensor = self._soft_supervision(real_A,0.7,0.3,0.25)
            real_msk_t,_ = self._soft_supervision(real_A_t,0.7,0.3,0.25)
            IoU = self.cal_iou_clap(real_msk,real_msk_t)
            loss_real = (CELP(real_B,real_msk.squeeze().long()).squeeze())
            loss_real = (IoU*loss_real.mean((1,2))).mean()
            loss = loss_real
            return loss, loss_real,IoU.mean()

    def random_drop(self,drop_block,image):
        Mask  = torch.ones_like(image)
        a = [x for x in range(16)]
        random.shuffle(a)
        a = a[:drop_block]
        w = [x%4 for x in a]
        h = [x/4 for x in a]
        for i in range(4):
            w_low = w[i]*32
            w_high = w_low+32
            h_low = int(h[i]*32)
            h_high = int(h_low+32)
            Mask[:,w_low:w_high,h_low:h_high]=0
        return Mask