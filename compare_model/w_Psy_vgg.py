import torch.nn as nn
from numpy import record
import torchvision.models as vmodels
from utils.postprocessing import *
import torch.nn.functional as F
from utils.utils import acc_counter,loss_counter

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'vggcam16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512, 'N'],
    'vggcam19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'N', 512, 512, 512, 512, 'N'],
}


class Psy_o1(nn.Module):
    def __init__(self, vggconfig, tftypes=['rotation', 'translation', 'shear', 'hflip'], tfnums=[4, 3, 3, 2], with_cam=False):
        super(Psy_o1, self).__init__()
        # parameters setting
        use_bn = ('bn' in vggconfig)
        features = make_layers(cfg[vggconfig[:8]], batch_norm=use_bn)
        self.num_cls = tfnums
        self.with_cam = with_cam
        print('USE TF IN NET:\t', tftypes)
        print("USE TF NUMS:\t", self.num_cls)

        chinter = 512
        #根据数据集几何变换组合生成相应结构的网路
        self.blocks = self._geo_blocks(tftypes,chinter)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self._initialize_weights()

        self.features = features

    def forward(self,x):
        logits = []
        cams = []
        x = self.features(x)
        # x = self.inter(x)
        chatt = x.mean(1)
        for block in self.blocks:
            cam = block(x)
            cams.append(cam)
            logit = self.pool(cam).squeeze()
            logits.append(logit)
        cams.append(chatt)
        if self.training == False:
            return logits, cams[-1]
        return logits, cams

    def get_loss(self,c_logit,labels):
        c_loss = torch.tensor([0.0]).cuda(non_blocking=True)
        record_loss = []
        for logitidx in range(len(c_logit)):
            tmp_loss = F.cross_entropy(c_logit[logitidx], labels[logitidx])
            record_loss.append(tmp_loss)
            c_loss = c_loss + tmp_loss
        record_loss.insert(0,c_loss)
        return c_loss,record_loss

    def loss_init(self):
        self.loss_ct =[]
        for i in range(3):
            self.loss_ct.append(loss_counter())
        for i in range(2):
            self.loss_ct.append(acc_counter())

    def loss_count(self, record_loss,c_logit,labels):
        for i in range(3):
            self.loss_ct[i].count(record_loss[i])
        for i in range(2):
            # ipdb.set_trace()
            self.loss_ct[i+3].count(c_logit[i].argmax(1).cpu().numpy(),labels[i].cpu().numpy())

    def loss_reset(self):
        for i in range(5):
            self.loss_ct[i].reset()

    def loss_output(self):
        loss = self.loss_ct[0]()
        loss_PST = self.loss_ct[1]()
        loss_RoT = self.loss_ct[2]()
        acc_PST = self.loss_ct[3]()
        acc_RoT = self.loss_ct[4]()
        output = ('loss:{:.4f},loss_PST:{:.4f},loss_RoT:{:.4f},acc_PST:{:.4f},'+\
                    'acc_RoT:{:.4f}')\
            .format(loss,loss_PST,loss_RoT,acc_PST,acc_RoT)
        return output


    def _geo_blocks(self,tftypes,chinter):
        blocks = nn.ModuleList()
        if 'odd' in tftypes:
            print('APPEND ODD')
            self.odd = nn.Conv2d(chinter, self.num_cls[5], 1, 1)
            blocks.append(self.odd)
        if 'rotation' in tftypes:
            print('APPEND ROTATION')
            self.rotation = nn.Conv2d(chinter, self.num_cls[0], 1, 1)
            blocks.append(self.rotation)
        if 'translation' in tftypes:
            print('APPEND TRANSLATION')
            self.translation = nn.Conv2d(chinter, self.num_cls[1], 1, 1)
            blocks.append(self.translation)
        if 'shear' in tftypes:
            print('APPEND SHEAR')
            self.shear = nn.Conv2d(chinter, self.num_cls[2], 1, 1)
            blocks.append(self.shear)
        if 'hflip' in tftypes:
            print('APPEND HFLIP')
            self.hflip = nn.Conv2d(chinter, self.num_cls[3], 1, 1)
            blocks.append(self.hflip)
        if 'scale' in tftypes:
            print('APPEND SCALE')
            self.scale = nn.Conv2d(chinter, self.num_cls[4], 1, 1)
            blocks.append(self.scale)

        # blocks = nn.ModuleList()
        # if 'odd' in tftypes:
        #     print('APPEND ODD')
        #     self.odd = nn.Conv2d(chinter, self.num_cls[7], 1, 1)
        #     blocks.append(self.odd)
        # if 'rotation' in tftypes:
        #     print('APPEND ROTATION')
        #     self.rotation = nn.Conv2d(chinter, self.num_cls[0], 1, 1)
        #     blocks.append(self.rotation)
        # if 'translation' in tftypes:
        #     print('APPEND TRANSLATION')
        #     self.translation = nn.Conv2d(chinter, self.num_cls[1], 1, 1)
        #     blocks.append(self.translation)
        # if 'shear' in tftypes:
        #     print('APPEND SHEAR')
        #     self.shear = nn.Conv2d(chinter, self.num_cls[2], 1, 1)
        #     blocks.append(self.shear)
        # if 'hflip' in tftypes:
        #     print('APPEND HFLIP')
        #     self.hflip = nn.Conv2d(chinter, self.num_cls[3], 1, 1)
        #     blocks.append(self.hflip)
        # if 'scale' in tftypes:
        #     print('APPEND SCALE')
        #     self.scale = nn.Conv2d(chinter, self.num_cls[4], 1, 1)
        #     blocks.append(self.scale)
        # if 'vflip' in tftypes:
        #     print('APPEND VFLIP')
        #     self.vflip = nn.Conv2d(chinter, self.num_cls[5], 1, 1)
        #     blocks.append(self.vflip)
        # if 'vtranslation' in tftypes:
        #     print('APPEND VTRANSLATION')
        #     self.vtranslation = nn.Conv2d(chinter, self.num_cls[6], 1, 1)
        #     blocks.append(self.vtranslation)
        return blocks

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class Psy_o3(Psy_o1):
    def __init__(self, vggconfig, tftypes=['rotation', 'translation', 'shear', 'hflip'], tfnums=[4, 3, 3, 2], with_cam=False):
        super(Psy_o3, self).__init__(vggconfig, tftypes,tfnums)
        self.features_1 = self.features[0:40]
        self.features_2 = self.features[40:]

    def forward(self, x):
        logits = []
        cams = []
        relu1 = self.features_1(x)
        x = self.features_2(relu1)
        # x = self.inter(x)
        chatt = x.mean(1)
        for block in self.blocks:
            cam = block(x)
            cams.append(cam)
            logit = self.pool(cam).squeeze()
            logits.append(logit)
        cams.append(relu1.mean(1))
        cams.append(chatt)
        if self.training == False:
            attmap1 = cams[-1]
            attmap2 = cams[-2]
            attmap = norm_att_map(attmap1)
            a = torch.mean(attmap, dim=(1, 2), keepdim=True)
            attmap = (attmap > a).float()
            attmap2 = norm_att_map(attmap2)
            a2 = torch.mean(attmap2, dim=(1, 2), keepdim=True)
            attmap2 = (attmap2 > a2).float()#做了个阈值筛选
            attmap = F.interpolate(attmap.unsqueeze(dim=1), (attmap2.size(1), attmap2.size(2)), mode='nearest').squeeze()
            attmap = attmap2 * attmap
            return logits,attmap
        return logits, cams


def make_layers(cfg, in_ch=3, batch_norm=False):
    layers = []
    in_channels = in_ch
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v, momentum=0.001), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# def model(vggconfig, tftypes, tfnums, **kwargs):
#     use_bn = ('bn' in vggconfig)
#     model = VGGTF(features=make_layers(cfg[vggconfig[:8]], batch_norm=use_bn),
#                       tftypes=tftypes, tfnums=tfnums)
    
#     return model