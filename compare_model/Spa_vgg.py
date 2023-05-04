import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import random
import numpy as np
import os
import ipdb
from utils.utils import to_image, acc_counter,loss_counter
from torch.nn import CrossEntropyLoss 
__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'model'
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
    def __init__(self, features, num_classes=1000, cnvs=(10,17,24), args=None):
        super(VGG, self).__init__()
        self.conv1_2 = nn.Sequential(*features[:cnvs[0]])
        self.conv3 = nn.Sequential(*features[cnvs[0]:cnvs[1]])
        self.conv4 = nn.Sequential(*features[cnvs[1]:cnvs[2]])
        self.conv5 = nn.Sequential(*features[cnvs[2]:-1])
        self.conv5_add = nn.Sequential(
            nn.Conv2d(512, 512,3, padding=1),
            nn.ReLU(inplace=True))
        self.fmp = features[-1]  # final max pooling
        self.num_classes = num_classes
        self.args = args

        self.cls = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1),  # fc6
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, dilation=1),  # fc7
            nn.ReLU(True),
            nn.Conv2d(1024, self.num_classes, kernel_size=1, padding=0)
        )

        self._initialize_weights()

        # loss function
        self.loss_cross_entropy = F.cross_entropy
        self.loss_bce = F.binary_cross_entropy_with_logits
        self.nll_loss = F.nll_loss

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



    def forward(self, x, label=torch.tensor([1]), return_cam = False ):
        x = self.conv1_2(x)
        #sc_2 = None
        #sc_2_so = None
        #if scg_flag and '2' in self.args.scg_blocks :
        #    sc_2, sc_2_so = self.hsc(x, fo_th=self.args.scg_fosc_th,
        #                                   so_th=self.args.scg_sosc_th,
        #                                   order=self.args.scg_order)

        feat_3 = self.conv3(x)
        #sc_3 = None
        #sc_3_so = None
        #if scg_flag and '3' in self.args.scg_blocks :
        #    sc_3, sc_3_so = self.hsc(feat_3, fo_th=self.args.scg_fosc_th,
        #                                   so_th=self.args.scg_sosc_th,
        #                                   order=self.args.scg_order)

        feat_4 = self.conv4(feat_3)
        self.feat4= feat_4
        sc_4 = None
        sc_4_so = None
        if self.training == False and '4' in self.args.scg_blocks:
            sc_4, sc_4_so = self.hsc(feat_4, fo_th=self.args.scg_fosc_th,
                                    so_th=self.args.scg_sosc_th, order=self.args.scg_order)

        feat_5 = self.conv5(feat_4)
        self.feat5= feat_5
        sc_5 = None
        sc_5_so = None
        if self.training == False and '5' in self.args.scg_blocks:
            sc_5, sc_5_so = self.hsc(feat_5, fo_th=self.args.scg_fosc_th,
                                                  so_th=self.args.scg_sosc_th,
                                                  order=self.args.scg_order)
        cls_map = self.cls(feat_5)
        self.cls_map = cls_map
        logits = torch.mean(torch.mean(cls_map, dim=2), dim=2)
        if self.training == False:
            CAM = self.get_atten_map(cls_map,torch.tensor([torch.argmax(logits).item()]),True)
            if self.args.scg == True:
                CAM_scg = self.cam_scg(CAM, (sc_4,sc_5),(sc_4_so,sc_5_so))
                return torch.tensor(CAM_scg) ,[logits,] 
            
            return CAM.detach().cpu(),[logits,] 

        return [logits,]

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
        HSC = sc_maps[-1]+sc_maps[-2]
        return HSC

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

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        #--------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

    def hsc(self, f_phi, fo_th=0.1, so_th=0.1, order=2):
        """
        Calculate affinity matrix and update feature.
        :param feat:
        :param f_phi:
        :param fo_th:
        :param so_weight:
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
        for _ in range(order-1):
            non_local_cos_ho = torch.matmul(non_local_cos_ho, non_local_cos)
            # non_local_cos_ho[:, torch.arange(h * w), torch.arange(w * h)] = 0
            non_local_cos_ho = non_local_cos_ho / (torch.sum(non_local_cos_ho, dim=1, keepdim=True) + 1e-10)
        # non_local_cos_ho = non_local_cos_ho - torch.min(non_local_cos_ho, dim=1, keepdim=True)[0]
        #non_local_cos_ho = non_local_cos_ho / (torch.max(non_local_cos_ho, dim=1, keepdim=True)[0] + 1e-10)
        non_local_cos_ho[non_local_cos_ho < so_th] = 0
        return  non_local_cos_fo, non_local_cos_ho


    def get_loss(self, logits, gt_child_label, epoch=0, ram_start=20):
        
        CEL = CrossEntropyLoss()
        loss = CEL(logits[0],gt_child_label.long())
        if self.args.ram and epoch >= ram_start:
            ra_loss = self.get_ra_loss(self.cls_map, gt_child_label, self.args.ram_th_bg, self.args.ram_bg_fg_gap)
            loss += self.args.ra_loss_weight * ra_loss
        else:
            ra_loss = torch.zeros_like(loss)

        return [loss]

    def get_ra_loss(self, logits, label, th_bg=0.3, bg_fg_gap=0.0):
        n, _, _, _ = logits.size()
        cls_logits = F.softmax(logits, dim=1)
        var_logits = torch.var(cls_logits, dim=1)
        norm_var_logits = self.normalize_feat(var_logits)

        bg_mask = (norm_var_logits < th_bg).float()
        fg_mask = (norm_var_logits > (th_bg + bg_fg_gap)).float()
        cls_map = logits[torch.arange(n), label.long(), ...]
        cls_map = torch.sigmoid(cls_map)

        ra_loss = torch.mean(cls_map * bg_mask + (1 - cls_map) * fg_mask)
        return ra_loss

    def normalize_feat(self,feat):
        n, fh, fw = feat.size()
        feat = feat.view(n, -1)
        min_val, _ = torch.min(feat, dim=-1, keepdim=True)
        max_val, _ = torch.max(feat, dim=-1, keepdim=True)
        norm_feat = (feat - min_val) / (max_val - min_val + 1e-15)
        norm_feat = norm_feat.view(n, fh, fw)

        return norm_feat

    def get_cls_maps(self):
        return F.relu(self.cls_map)
    def get_loc_maps(self):
        return torch.sigmoid(self.loc_map)

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

def make_layers(cfg, dilation=None, batch_norm=False, instance_norm=False, inl=False):
    layers = []
    in_channels = 3
    for v, d in zip(cfg, dilation):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'L':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d, dilation=d)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            elif instance_norm and v <256 and v>64:
                layers += [conv2d, nn.InstanceNorm2d(v, affine=inl), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    # 'D_deeplab': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'O': [64, 64, 'L', 128, 128, 'L', 256, 256, 256, 'L', 512, 512, 512, 'L', 512, 512, 512, 'L']
}

dilation = {
    'D_deeplab': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'N', 2, 2, 2, 'N'],
    'D1': [1, 1, 'M', 1, 1, 'M', 1, 1, 1, 'M', 1, 1, 1, 'N', 1, 1, 1, 'N']
}

cnvs= {'O': (10,7,7), 'OI':(12,7,7)}

def model(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet

    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    """

    layers = make_layers(cfg['D1'], dilation=dilation['D1'])
    cnv = np.cumsum(cnvs['O'])
    model = VGG(layers, cnvs=cnv, **kwargs)
    if pretrained:
        pre2local_keymap = [('features.{}.weight'.format(i), 'conv1_2.{}.weight'.format(i)) for i in range(10)]
        pre2local_keymap += [('features.{}.bias'.format(i), 'conv1_2.{}.bias'.format(i)) for i in range(10)]
        pre2local_keymap += [('features.{}.weight'.format(i + 10), 'conv3.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 10), 'conv3.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.weight'.format(i + 17), 'conv4.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 17), 'conv4.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.weight'.format(i + 24), 'conv5.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 24), 'conv5.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap = dict(pre2local_keymap)


        model_dict = model.state_dict()
        print(model_dict.keys())
        pretrained_file = kwargs['args'].restore_from
        if os.path.isfile(pretrained_file):
            pretrained_dict = torch.load(pretrained_file)
            print('load pretrained model from {}'.format(pretrained_file))
        else:
            pretrained_dict = model_zoo.load_url(model_urls['vgg16'])
            print('load pretrained model from {}'.format(model_urls['vgg16']))
        # 0. replace the key
        pretrained_dict = {pre2local_keymap[k] if k in pre2local_keymap.keys() else k: v for k, v in
                           pretrained_dict.items()}
        # *. show the loading information
        for k in pretrained_dict.keys():
            if k not in model_dict:
                print('Key {} is removed from vgg16'.format(k))
        print(' ')
        for k in model_dict.keys():
            if k not in pretrained_dict:
                print('Key {} is new added for DA Net'.format(k))
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    model(True)
