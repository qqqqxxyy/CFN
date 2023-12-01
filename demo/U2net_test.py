#-*- coding: UTF-8 -*-
from random import randint
from re import I, U
from torchvision import transforms
import sys
#添加检索路径
sys.path.append('..')
from UNet.unet_model import UNet
#SegmentationInference, Threshold,resize_min_edge
import os

import argparse
import json
import torch
import cv2
from torchvision.transforms import ToPILImage, ToTensor, Resize
from compare_model import Spa_vgg,CAM_vgg,I2c_vgg,ACoL_vgg
from compare_model.inceptionv3 import inceptionv3_ACoL,inceptionv3_CAM,inceptionv3_SPA
import numpy as np
#from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import ipdb
import torch.nn.functional as F
from utils.io_util import Info_Record,formate_output,Pretrained_Read,load_path_file
from utils.postprocessing import *
from utils.utils import to_image,IoU_manager,area_counter
from utils.visual_util import visualizer
from UNet.unet_cam import *
from UNet.u2net import *
from UNet.unet_ssol import *
from gan_mask_gen import MaskGenerator, MaskSynthesizing, it_mask_gen #通过gan网络生成mask
from data import ObjLocDataset,Trans,TransBbox,SegFileDataset,OriDataset,ObjMaskDataset,TransMask
from data import CombData_pre as CombData,Random_noise
from metrics import *
#model_metrics, IoU, accuracy, F_max,Localization,mask2IoU,cal_IoU
from BigGAN.gan_load import make_big_gan
from tqdm import tqdm
from ptflops import get_model_complexity_info
import pandas as pd


class TestParams(object):

    def __init__(self):
        #固定参数
        self.batch_size= 1    
        self.model = ""
        self.dataset = ""
        self.n_steps=None
        self.phase = 1
        self.val_property_dir = ""
        self.image_root_dir = None
        self.image_property_dir = None
        self.latent_shift_r = 5.0
        self.synthezing = MaskSynthesizing.LIGHTING
        self.mask_size_up = 0.5
        self.connected_components = True
        self.maxes_filter = True
        self.syn_norm=True
        self.save_iou=False
        self.model_weight=""
        self.length=50
        #读入flexible_config的文件中的参数
        self.flexible_config_path=""
        self.flexible_config_choice=""
        parser = argparse.ArgumentParser(description='Test Parameters')
        parser.add_argument('-cfgp','--flexible_config_path', type=str, default=None)
        parser.add_argument('-cfgc','--flexible_config_choice', type=str, default=None)
        args = parser.parse_args()       
        for key, val in args.__dict__.items():
            if val is not None:
                self.__dict__[key] = val
        self._read_from_config(self.flexible_config_path, self.flexible_config_choice)
        #读入base_config文件中的参数
        PATH_FILE = load_path_file()
        self.root_path=PATH_FILE["root_path"]
        self.dataset_path=PATH_FILE["dataset_path"]
        #补全model_weight,val_property_dir路径
        self.model_weight=os.path.join(self.root_path,'weight',self.model_weight)
        self.val_property_dir=os.path.join(self.dataset_path,self.dataset,'data_annotation',self.val_property_dir)
        #连接数据路径
        self._complete_formed_paths(self.dataset_path,'mask_root_dir')
        self._complete_formed_paths(self.dataset_path,'val_root_dir')
        self._complete_paths(self.dataset_path,'val_property_dir')
        


    def _read_from_config(self,config_path,config_choice="default"):
        with open(config_path,'r') as f:
            cfg=json.load(f)[config_choice]
        for key,val in cfg.items():
            if val is not None:
                self.__dict__[key]=val


    def _complete_paths(self,root_path,abspath):
        abss = self.__dict__[abspath]
        self.__dict__[abspath] = os.path.join(root_path,abss)

    def _complete_formed_paths(self,root_path,abspath):
        dataset = self.__dict__['dataset']
        if abspath == 'mask_root_dir':
            self.__dict__[abspath] = os.path.join(root_path,dataset,'segmentation')
        if abspath == 'image_root_dir':
            self.__dict__[abspath] = os.path.join(root_path,dataset,'data')
        if abspath == 'val_root_dir':
            self.__dict__[abspath] = os.path.join(root_path,dataset,'data')
        if abspath == 'image_property_dir':
            self.__dict__[abspath] = os.path.join(root_path,dataset,'data_annotation/train_list.json')

def main():

    param = TestParams()
    print('run test with parameter:')
    formate_output(param.__dict__)
    info_record = Info_Record('test')
    info_record.record(param.__dict__,'both')
    init_seq = '{0} experment No: {1}  time : {2}'.format(\
            info_record.mode,info_record.id,info_record.time_sequence)
    print(init_seq)


    check_unet(param,info_record)

    info_record.record('test complete as plan!')



def check(param,info_record):

    target_ds = ObjLocDataset(param.val_property_dir,param.val_property_dir,
                tsfm=transforms.Compose([ transforms.Resize((128,128)),
                    transforms.ToTensor(), Trans(['Normalize','standard']) ]),
                tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(128,128)]) ]) )

    unet = eval(param.model)().cuda().eval()
    unet.load_state_dict(torch.load(param.model_weight, map_location='cpu'))
    unet.cuda().eval()

    thr_scale = [0.01,0.30,0.02]
    IoU_man = IoU_manager(thr_scale)    
    for t in tqdm(range(param.n_steps)):
        _, image,clas, bbox= target_ds[t]
        d1,d2,d3,d4,d5,d6,d7 = unet(image.unsqueeze(0).cuda())
        pred = d1[:,0,:,:]
        # pred = (pred-pred.min())/(pred.max()-pred.min())
        # ipdb.set_trace()
        IoU_man.update(pred,bbox)

    info_dict = IoU_man.acc_output(disp_form=True)
    info_record.record(thr_scale,'log_detail')
    info_record.formate_record(info_dict,'log_detail')
    print(info_dict)

def cal_iou(map_a,map_b):
    '''map_a and map_b both:
    [N,1,128,128]
    '''
    map_a, map_b = map_a.to(torch.bool), map_b.to(torch.bool)
    intersection = torch.sum(map_a * (map_a == map_b), dim=[-1, -2]).squeeze()
    union = torch.sum(map_a + map_b, dim=[-1, -2]).squeeze()
    return (intersection.to(torch.float) / (union+1e-8))

def _model_load(model, pretrained_dict):
    '''this function can help to load the consistent part and ignore the rest part
    if parameters in checkpoint not totally fit model (this happens when train CAM-based 
    model with imagenet pretrained backbone)
    '''
    model_dict = model.state_dict()
    pretrained_dict_b = {}
    for key in pretrained_dict.keys():
        str_head = '.'
        key_b = key.split('.')
        if 'outc' in key_b[0]:
            key_bb = 'B_'+key_b[0]+'.'+str_head.join(key_b[1:])
            key_a = 'A_'+key_b[0]+'.'+str_head.join(key_b[1:])
            pretrained_dict_b.update({key_bb:pretrained_dict[key]}) 
            pretrained_dict_b.update({key_a:pretrained_dict[key]}) 
        else:
            key_b = key_b[0]+'_B.'+str_head.join(key_b[1:])
            pretrained_dict_b.update({key_b:pretrained_dict[key]}) 
    pretrained_dict.update(pretrained_dict_b)


    print ("Weights cannot be loaded:")
    print ([k for k in model_dict.keys() if k not in pretrained_dict.keys()])
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)  

def _model_load2(model, pretrained_dict):
    '''this function can help to load the consistent part and ignore the rest part
    if parameters in checkpoint not totally fit model (this happens when train CAM-based 
    model with imagenet pretrained backbone)
    '''
    model_dict = model.state_dict()
    # model_dict_keys = [v.replace('module.', '') for v in model_dict.keys() if v.startswith('module.')]
    if list(model_dict.keys())[0].startswith('module.')  and  \
        list(pretrained_dict.keys())[0].startswith('module.')== False :
            pretrained_dict = {'module.'+k: v for k, v in pretrained_dict.items()}


    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    print ("Weights cannot be loaded:")
    print ([k for k in model_dict.keys() if k not in pretrained_dict.keys()])
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)  

def check_unet(param,info_record):
    target_ds = ObjLocDataset(param.val_root_dir,param.val_property_dir,
                tsfm=transforms.Compose([ transforms.Resize((128,128)),
                # tsfm=transforms.Compose([ transforms.Resize(128),
                    transforms.ToTensor(), Trans(['Normalize','standard']) ]),
                tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(128,128)]) ]) )

    unet = eval(param.model)().cuda().eval()
    # unet = torch.nn.DataParallel(unet, range(1))
    pretrain_info = Pretrained_Read()
    pretrain_info.add_info_dict(param.model_weight)

    try:
        unet.load_state_dict(pretrain_info.model_weight)
    except:
        try:
            unet.module.load_state_dict(pretrain_info.model_weight)
        except:
            _model_load(unet,pretrain_info.model_weight)


    unet.cuda().eval()
    unet.phase= param.phase
    # unet.initialize_weights('B')
    thr_scale = [0.01,0.96,0.02]

    IoU_man = IoU_manager(thr_scale,mark='bbox',save_iou=param.save_iou) 
    # area_count = area_counter()
    if param.n_steps == None:
        param.n_steps = len(target_ds)
    for t in tqdm(range(param.n_steps)):
        _, image,clas, bbox,size= target_ds[t]
        mask_pre,cls = unet(image.unsqueeze(0).cuda())
        # mask_pre = mask_pre.detach().cpu().detach().numpy()
        # mask_pre = (mask_pre-mask_pre.min())/(mask_pre.max()-mask_pre.min())
        _, iou = IoU_man.update(mask_pre,bbox)
        
        # mask_pre = mask_pre>0.5
        # cor_pre_list = mask2cor(mask_pre.squeeze(),0.4)
        # area_count.count(cor_pre_list,bbox[0])

    # print(area_count())
    info_dict = IoU_man.acc_output(disp_form=True)
    if param.save_iou == True:
        IoU_man.save_iou_list(id = info_record.id,dataset_name=param.dataset)


    info_record.record(thr_scale,'log_detail')
    info_record.formate_record(info_dict,'log_detail')
    print(info_dict)


def check_violin(param,info_record):
    target_ds = ObjLocDataset(param.val_root_dir,param.val_property_dir,
                tsfm=transforms.Compose([ transforms.Resize((128,128)),
                # tsfm=transforms.Compose([ transforms.Resize(128),
                    transforms.ToTensor(), Trans(['Normalize','standard']) ]),
                tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(128,128)]) ]) )

    unet = eval(param.model)().cuda().eval()
    # unet = torch.nn.DataParallel(unet, range(1))
    pretrain_info = Pretrained_Read()
    pretrain_info.add_info_dict(param.model_weight)
    #data stuff
    Id = info_record.id
    save_paths = '/home/qxy/Desktop/beta/results/Figure/{}_360_violin.xlsx'.format(Id)
    IoU_list = []
    try:
        unet.load_state_dict(pretrain_info.model_weight)
    except:
        try:
            unet.module.load_state_dict(pretrain_info.model_weight)
        except:
            _model_load(unet,pretrain_info.model_weight)


    unet.cuda().eval()
    unet.phase= param.phase
    # unet.initialize_weights('B')
    thr_scale = [0.30,0.31,0.02]

    IoU_man = IoU_manager(thr_scale,mark='bbox') 
    # area_count = area_counter()
    if param.n_steps == None:
        param.n_steps = len(target_ds)
    for t in tqdm(range(param.n_steps)):
        _, image,clas, bbox= target_ds[t]
        mask_pre,cls = unet(image.unsqueeze(0).cuda())
        # mask_pre = mask_pre.detach().cpu().detach().numpy()
        # mask_pre = (mask_pre-mask_pre.min())/(mask_pre.max()-mask_pre.min())
        _, iou = IoU_man.update(mask_pre,bbox)
        if iou[0]>0.5:
            IoU_list.append(iou[0])
        # mask_pre = mask_pre>0.5
        # cor_pre_list = mask2cor(mask_pre.squeeze(),0.4)
        # area_count.count(cor_pre_list,bbox[0])

    # print(area_count())
    info_dict = IoU_man.acc_output(disp_form=True)
    print(np.median(np.array(IoU_list)))
    df = pd.DataFrame({'iou':IoU_list})
    df.to_excel(save_paths,index=False,sheet_name='sheet1')
    
    info_record.record(thr_scale,'log_detail')
    info_record.formate_record(info_dict,'log_detail')
    print(info_dict)

def check_compare(param,info_record):
    target_ds = ObjLocDataset(param.val_root_dir,param.val_property_dir,
                tsfm=transforms.Compose([ transforms.Resize((224,224)),
                # tsfm=transforms.Compose([ transforms.Resize(128),
                    transforms.ToTensor(), Trans(['Normalize','standard']) ]),
                tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(28,28)]) ]) )
    
    if 'CUB'in param.val_root_dir:
        unet = eval(param.model).model(num_classes=200,args=param)
    else:
        unet = eval(param.model).model(num_classes=1000,args=param)

    # unet = torch.nn.DataParallel(unet, range(1))
    pretrain_info = Pretrained_Read()
    pretrain_info.add_info_dict(param.model_weight)
    unet.cuda().eval()
    unet = torch.nn.DataParallel(unet.train().cuda(), range(1))
    unet.module.eval()
    try:
        unet.load_state_dict(pretrain_info.model_weight)
    except:
        try:
            unet.module.load_state_dict(pretrain_info.model_weight)
        except:
            _model_load2(unet,pretrain_info.model_weight)



    unet.phase= param.phase
    # unet.initialize_weights('B')
    thr_scale = [0.1,0.5,0.01]

    IoU_man = IoU_manager(thr_scale,mark='bbox') 
    # area_count = area_counter()
    if param.n_steps == None:
        param.n_steps = len(target_ds)
    for t in tqdm(range(param.n_steps)):
        _, image,clas, bbox= target_ds[t]
        mask_pre,cls = unet(image.unsqueeze(0).cuda())
        # mask_pre = mask_pre.detach().cpu().detach().numpy()
        # mask_pre = (mask_pre-mask_pre.min())/(mask_pre.max()-mask_pre.min())
        _, iou = IoU_man.update(mask_pre,bbox)
        ipdb.set_trace()
        # mask_pre = mask_pre>0.5
        # cor_pre_list = mask2cor(mask_pre.squeeze(),0.4)
        # area_count.count(cor_pre_list,bbox[0])

    # print(area_count())
    info_dict = IoU_man.acc_output(disp_form=True)

    info_record.record(thr_scale,'log_detail')
    info_record.formate_record(info_dict,'log_detail')
    print(info_dict)

def check_compare_vis(param,info_record):
    target_ds = ObjLocDataset(param.val_root_dir,param.val_property_dir,
                tsfm=transforms.Compose([ transforms.Resize((224,224)),
                # tsfm=transforms.Compose([ transforms.Resize(128),
                    transforms.ToTensor(), Trans(['Normalize','standard']) ]),
                tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(28,28)]) ]) )
    
    if 'CUB'in param.val_root_dir:
        unet = eval(param.model).model(num_classes=200,args=param)
    else:
        unet = eval(param.model).model(num_classes=1000,args=param)

    # unet = torch.nn.DataParallel(unet, range(1))
    pretrain_info = Pretrained_Read()
    pretrain_info.add_info_dict(param.model_weight)
    unet.cuda().eval()
    unet = torch.nn.DataParallel(unet.train().cuda(), range(1))
    unet.module.eval()
    try:
        unet.load_state_dict(pretrain_info.model_weight)
    except:
        try:
            unet.module.load_state_dict(pretrain_info.model_weight)
        except:
            _model_load2(unet,pretrain_info.model_weight)



    unet.phase= param.phase
    # unet.initialize_weights('B')
    thr_scale = [0.1,0.5,0.01]

    IoU_man = IoU_manager(thr_scale,mark='bbox')
    vis = visualizer()
    save_paths = 'XXX'
        
        
        
    # area_count = area_counter()
    if param.n_steps == None:
        param.n_steps = len(target_ds)
    for t in tqdm(range(param.n_steps)):
        _, image,clas, bbox= target_ds[t]
        mask_pre,cls,feature_map = unet(image.unsqueeze(0).cuda())
        save_pathss = '/home/qxy/Desktop/beta/results/pictures/XXX/feature_map.pt'
        torch.save(feature_map, save_pathss)
        mask_pre = F.interpolate(mask_pre.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=True)
        vis.save_htmp(mask_pre,image= image,paths=[save_paths,'{}_0_{}.png'.format(t,param.model)])
        print(mask_pre.shape)
        
def check_unet_rot(param,info_record):
    target_ds = ObjLocDataset(param.val_root_dir,param.val_property_dir,
                tsfm=transforms.Compose([ transforms.Resize((128,128)),
                # tsfm=transforms.Compose([ transforms.Resize(128),
                    transforms.ToTensor(), Trans(['Normalize','standard']) ]),
                tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(128,128)]) ]) )

    unet = eval(param.model)().cuda().eval()
    # unet = torch.nn.DataParallel(unet, range(1))
    pretrain_info = Pretrained_Read()
    pretrain_info.add_info_dict(param.model_weight)

    try:
        unet.load_state_dict(pretrain_info.model_weight)
    except:
        try:
            unet.module.load_state_dict(pretrain_info.model_weight)
        except:
            _model_load(unet,pretrain_info.model_weight)


    unet.cuda().eval()
    unet.phase= param.phase
    # unet.initialize_weights('B')
    thr_scale = [0.50,0.51,0.01]
    a_08_10 = [0,0]
    a_06_08 = [0,0]
    a_04_06 = [0,0]
    a_02_04 = [0,0]
    a_00_02 = [0,0]

    IoU_man = IoU_manager(thr_scale,mark='bbox') 
    # area_count = area_counter()
    if param.n_steps == None:
        param.n_steps = len(target_ds)
    for t in tqdm(range(param.n_steps)):
        _, image,clas, bbox= target_ds[t]
        mask_pre,cls = unet(image.unsqueeze(0).cuda())
        # mask_pre = mask_pre.detach().cpu().detach().numpy()
        # mask_pre = (mask_pre-mask_pre.min())/(mask_pre.max()-mask_pre.min())
        _, iou = IoU_man.update(mask_pre,bbox)
        
        rot_180 = transforms.RandomRotation(degrees=(180, 180), expand=False)   
        image_t = rot_180(image)
        mask_t_pre ,_ = unet(image_t.unsqueeze(0).cuda())
        mask_t_pre = rot_180(mask_t_pre)
        mask_pre = mask_pre>0.5
        mask_t_pre = mask_t_pre>0.5
        geo_iou = cal_iou(mask_pre,mask_t_pre)

        if iou[0] >= 0.8:
            a_08_10[0]+=1
            a_08_10[1]+=geo_iou
        elif iou[0] >= 0.6:
            a_06_08[0]+=1
            a_06_08[1]+=geo_iou
        elif iou[0] >= 0.4:
            a_04_06[0]+=1
            a_04_06[1]+=geo_iou
        elif iou[0] >= 0.2:
            a_02_04[0]+=1
            a_02_04[1]+=geo_iou
        else:
            a_00_02[0]+=1
            a_00_02[1]+=geo_iou

    a_08_10[1] /= (a_08_10[0]+1e-8)
    a_08_10[0] /= param.n_steps
    a_06_08[1] /= (a_06_08[0]+1e-8)
    a_06_08[0] /= param.n_steps
    a_04_06[1] /= (a_04_06[0]+1e-8)
    a_04_06[0] /= param.n_steps
    a_02_04[1] /= (a_02_04[0]+1e-8)
    a_02_04[0] /= param.n_steps
    a_00_02[1] /= (a_00_02[0]+1e-8)
    a_00_02[0] /= param.n_steps
        # cor_pre_list = mask2cor(mask_pre.squeeze(),0.4)
        # area_count.count(cor_pre_list,bbox[0])

    # print(area_count())
    # info_dict = IoU_man.acc_output(disp_form=True)
    info_dict = {'a_08_10':a_08_10,'a_06_08':a_06_08,'a_04_06':a_04_06,'a_02_04':a_02_04\
        ,'a_00_02':a_00_02}
    info_record.record(thr_scale,'log_detail')
    info_record.formate_record(info_dict,'log_detail')
    print(info_dict)
    


def check_unet_mask(param,info_record):
    unet = eval(param.model)().cuda().eval()
    pretrain_info = Pretrained_Read()
    pretrain_info.add_info_dict(param.model_weight)
    
    try:
        unet.load_state_dict(pretrain_info.model_weight)
    except :
        _model_load(unet, pretrain_info.model_weight)

    unet.cuda().eval()
    unet.phase=param.phase

    target_ds = SegFileDataset(param,crop=False,size=(128,),val=True)
   

    thr_scale = [0.4,0.8,0.01]

    IoU_man = IoU_manager(thr_scale,mark='avg_bbox')    
    if param.n_steps == None:
        param.n_steps = len(target_ds)
    for t in tqdm(range(param.n_steps)):
        image,mask_orao= target_ds[t]
        mask_pre,cls = unet(image.unsqueeze(0).cuda())

        bbox = mask2cor(mask_orao,0.5)
        # mask_pre = (mask_pre-mask_pre.min())/(mask_pre.max()-mask_pre.min())
        IoU_man.update(mask_pre,bbox)

    info_dict = IoU_man.acc_output(disp_form=True)
    info_record.record(thr_scale,'log_detail')
    info_record.formate_record(info_dict,'log_detail')
    print(info_dict)

def check_syn(param,info_record):
    unet = eval(param.model)().cuda().eval()
    pretrain_info = Pretrained_Read()
    pretrain_info.add_info_dict(param.model_weight)
    
    try:
        unet.load_state_dict(pretrain_info.model_weight)
    except :
        _model_load(unet, pretrain_info.model_weight)

    unet.cuda().eval()
    unet.phase=param.phase


    G = make_big_gan(param.gan_weights).eval().cpu()
    bg_direction = torch.load(param.bg_direction)
    mask_postprocessing = [connected_components_filter]
    params = [G,bg_direction,mask_postprocessing,param]
    TrainOri = OriDataset(param.val_root_dir,param.val_property_dir,
                         tsfm=transforms.Compose([transforms.Resize((156,156)), 
                                    transforms.RandomCrop(128), 
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    Trans(['Normalize','standard']),
                                     ]))
    devision='synthetic'
    target_ds = CombData(TrainOri,MaskGenerator,devision,params)

    thr_scale = [0.10,0.50,0.02]

    IoU_man = IoU_manager(thr_scale,mark='bbox')    
    if param.n_steps == None:
        param.n_steps = len(target_ds)
    for t in tqdm(range(param.n_steps)):
        image,clas,mask_orao= target_ds[t]
        mask_pre,cls = unet(image.cuda())
        bbox = mask2cor(mask_orao.squeeze(),0.5)
        # mask_pre = (mask_pre-mask_pre.min())/(mask_pre.max()-mask_pre.min())
        IoU_man.update(mask_pre,bbox)

    info_dict = IoU_man.acc_output(disp_form=True)
    info_record.record(thr_scale,'log_detail')
    info_record.formate_record(info_dict,'log_detail')
    print(info_dict)

def check_train(model,dataloader,length=None):
    IoU_man = IoU_manager([0.01,0.90,0.02]) 
    for sample in tqdm(enumerate(dataloader)):
        idx,img,clas,bbox = sample[1]

        mask_pre,_ = model(img.cuda())
        # mask_pre = (mask_pre-mask_pre.min())/(mask_pre.max()-mask_pre.min())
        # mask_pre = (1.0 - torch.softmax(mask_pre, dim=1))[:, 0]
        IoU_man.update(mask_pre,bbox)
        if idx == length:
            break
    info_dict = IoU_man.acc_output(disp_form=True)
    return info_dict


def check_train_cls(model,dataloader,length=None):
    cls_counter = acc_counter()
    for sample in tqdm(enumerate(dataloader)):
        idx,img,clas,bbox = sample[1]
        cls_pre = model(img.cuda())
        cls_counter.count(cls_pre[:clas.shape[0],:].argmax(1).cpu().numpy(),clas.cpu().numpy())
    return cls_counter() 

def check_visualization(param, info_record):
    #real generation code
    target_ds = ObjLocDataset(param.val_root_dir,param.val_property_dir,
                tsfm=transforms.Compose([ transforms.Resize((128,128)),
                    transforms.ToTensor(), Trans(['Normalize','standard']) ]),
                tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(128,128)]) ]) )


    # #synthetic generation code
    # G = make_big_gan(param.gan_weights).eval().cpu()
    # bg_direction = torch.load(param.bg_direction)
    # mask_postprocessing = [connected_components_filter]
    # params = [G,bg_direction,mask_postprocessing,param]
    # devision = 'real'
    # # devision_lis = ['synthetic', 'split', 'split']
    # TrainOri = OriDataset(param.val_root_dir,param.val_property_dir,
    #             tsfm=transforms.Compose([ transforms.Resize((128,128)),
    #                 transforms.ToTensor(), Trans(['Normalize','0.5'])  ]) )

    # target_ds = CombData(TrainOri,MaskGenerator,devision,params)



    unet = eval(param.model)().cuda().eval()
    # unet = torch.nn.DataParallel(unet, range(1))
    pretrain_info = Pretrained_Read()
    pretrain_info.add_info_dict(param.model_weight)
    try:
        unet.load_state_dict(pretrain_info.model_weight)
    except:
        try:
            unet.module.load_state_dict(pretrain_info.model_weight)
        except:
            _model_load(unet,pretrain_info.model_weight)

    # unet.load_state_dict(pretrain_info.model_weight)

    unet.cuda().eval()
    unet.phase= param.phase
    vis = visualizer()    

    thr_scale = [0.30]#,0.51,0.02]

    for t in tqdm(range(0,100,1)):
        _, image,clas, bbox= target_ds[t] #real generator generation
        # image,clas, mask= target_ds[t] #synthetic generator generation
        # image = image.squeeze() #also syntheic code
        mask_pre,_ = unet(image.unsqueeze(0).cuda())

        avg = 0.50
        thr_scale = [avg]
        bbox_pre = mask2cor(mask_pre.squeeze(),thr_scale)
        # bbox_real = [int(x) for x in bbox[0]]
        # ipdb.set_trace()
        save_paths = 'XX'
        
        
        vis.save_htmp(mask_pre,image= image,paths=[save_paths,'{}_0.png'.format(t)])
        vis.save_graymp(mask_pre,paths=[save_paths,'{}_5.png'.format(t)])
        # vis.save_image(image,mask = mask_pre>avg,bbox = bbox_pre[0])
        vis.save_graymp(mask_pre>avg,paths=[save_paths,'{}_6.png'.format(t)])
        vis.save_image(image,mask = mask_pre>avg,paths=[save_paths,'{}_1.png'.format(t)])

        # vis.save_graymp(mask,paths=[save_paths,'{}__2.png'.format(t)])
        # vis.save_graymp(mask,image = image,paths=[save_paths,'{}__3.png'.format(t)])
        # vis.save_image(image,paths=[save_paths,'{}__1.png'.format(t)])
        
        rot_180 = transforms.RandomRotation(degrees=(180, 180), expand=False)   
        image_t = rot_180(image)
        mask_t_pre ,_ = unet(image_t.unsqueeze(0).cuda())
        
        # avg = 0.88
        thr_scale = [avg]
        bbox_pre = mask2cor(mask_pre.squeeze(),thr_scale)
        # bbox_real = [int(x) for x in bbox[0]]
        # ipdb.set_trace()
        vis.save_htmp(mask_t_pre,image= image_t,paths=[save_paths,'{}_2.png'.format(t)])
        vis.save_graymp(mask_t_pre,paths=[save_paths,'{}_7.png'.format(t)])
        # vis.save_image(image,mask = mask_pre>avg,bbox = bbox_pre[0])
        vis.save_graymp(mask_t_pre>avg,paths=[save_paths,'{}_8.png'.format(t)])
        # vis.save_image(image,mask = mask_pre>avg,bbox = bbox_pre[0])
        vis.save_image(image_t,mask = mask_t_pre>avg,paths=[save_paths,'{}_3.png'.format(t)])
        
        # vis.save_image(image,bbox = bbox_pre[0])

def check_syn_visualization(param,info_record):
    unet = eval(param.model)().cuda().eval()
    pretrain_info = Pretrained_Read()
    pretrain_info.add_info_dict(param.model_weight)
    
    try:
        unet.load_state_dict(pretrain_info.model_weight)
    except :
        _model_load(unet, pretrain_info.model_weight)

    unet.cuda().eval()
    unet.phase=param.phase


    # target_ds = ObjMaskDataset(param.val_root_dir,param.mask_root_dir,param.val_property_dir,
    #                      tsfm=transforms.Compose([TransMask(['Resize',(128,128)]), 
    #                                 TransMask(['ToTensor','None']),
    #                                 TransMask(['Normalize','standard']),
    #                                  ]))

    G = make_big_gan(param.gan_weights).eval().cpu()
    bg_direction = torch.load(param.bg_direction)
    mask_postprocessing = [connected_components_filter]
    params = [G,bg_direction,mask_postprocessing,param]
    devision = 'synthetic'
    # devision_lis = ['synthetic', 'split', 'split']
    TrainOri = OriDataset(param.val_root_dir,param.val_property_dir,
                         tsfm=transforms.Compose([transforms.Resize((156,156)), 
                                    transforms.RandomCrop(128), 
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    Trans(['Normalize','standard']),
                                     ]))
    target_ds = CombData(TrainOri,MaskGenerator,devision,params)

    thr_scale = [0.49,0.50,0.01]
    vis = visualizer()
    IoU_man = IoU_manager(thr_scale,mark='bbox')    

    for t in tqdm(range(0,50,1)):
        image,clas,mask = target_ds[t]
        image = image.squeeze()
        mask_pre,cls = unet(image.unsqueeze(0).cuda())
        mask = (Random_noise(mask.unsqueeze(0),0)>0.5).float()
        # pred = d1[:,0,:,:]
        # pred = (pred-pred.min())/(pred.max()-pred.min())
        bbox_pre = mask2cor(mask_pre.squeeze(),thr_scale)
        # bbox_pre,_ = IoU_man.update(mask_pre,bbox)
        bbox_real = mask2cor(mask.squeeze(),0.5)
        save_paths = 'syn_img_black'
        
        # vis.save_htmp(mask_pre,image= image,bbox_pre = bbox_pre[0], bbox_gt= bbox_real,\
        #     paths=[save_paths,'40_{}_0.png'.format(t)])
        # vis.save_image(image,mask = mask_pre>0.5,bbox = bbox_pre[0],\
        #     paths=[save_paths,'40_{}_1.png'.format(t)])
        vis.save_graymp(mask,paths=[save_paths,'{}_2.png'.format(t)])
        vis.save_graymp(mask,image = image,paths=[save_paths,'{}_3.png'.format(t)])
        vis.save_image(image,paths=[save_paths,'{}_1.png'.format(t)])

        IoU_man.update(mask_pre,bbox_real)

    info_dict = IoU_man.acc_output(disp_form=True)
    info_record.record(thr_scale,'log_detail')
    info_record.formate_record(info_dict,'log_detail')
    print(info_dict)

def check_wrong():
    param = TestParams()
    print('run test with parameter:')
    formate_output(param.__dict__)

    info_record = Info_Record('test')
    info_record.formate_record(param.__dict__,'both')

    unet = eval(param.model)().cuda().eval()
    unet.load_state_dict(torch.load(param.model_weight, map_location='cpu'))
    unet.cuda().eval()

    target_ds = SegFileDataset(param,crop=True,size=(128,128))
    mask_predictor = Threshold(unet,thr=0.5)
    vis = visualizer()
    mod = param.model_weight.split('/')[-1].split('.')[0]
    ct=0

    for i in tqdm(range(1000,2000)):
        image,mask = target_ds[i]
        # ipdb.set_trace()
        prediction = mask_predictor(image.unsqueeze(0).cuda())
        IoU,cor_real,cor_pre = mask2IoU(mask,prediction.squeeze())
        
        if IoU<0.5:
            ct += 1
            mask_pre,CAM_pre = unet(image.unsqueeze(0).cuda())

            mask_pre = (1.0 - torch.softmax(mask_pre, dim=1))[:, 0]

            vis.save_htmp(mask_pre,['121_w','{}_{}_1m.png'.format(i,mod)])
            vis.save_htmp(CAM_pre,['121_w','{}_{}_2c.png'.format(i,mod)])

            mask_pre = mask_pre>0.5
            vis.save_image(image,['121_w','{}_{}_0i.png'.format(i,mod)],
                            bbox = cor_pre[0],mask=mask_pre.squeeze())
            if ct == 50:
                model_metrics, IoU, accuracy, F_max,Localization,mask2IoU,cal_IoU
                return 

def check_mask_thr():
    param = TestParams()
    print('run test with parameter:')
    formate_output(param.__dict__)

    info_record = Info_Record('test')
    info_record.formate_record(param.__dict__,'both')
    init_seq = '{0} experment No: {1}  time : {2}'.format(\
            info_record.mode,info_record.id,info_record.time_sequence)
    print(init_seq)

    unet = eval(param.model)().cuda().eval()
    unet.load_state_dict(torch.load(param.model_weight, map_location='cpu'))
    unet.cuda().eval()

    target_ds = SegFileDataset(param,crop=True,size=(128,128))
    tar_ds2 = SegFileDataset(param,crop=False,size=128)
    val_ds = TestDataset(param.val_root_dir,param.val_property_dir)

    thr_scale = [0.50,0.70,0.01]
    start,end = 0,param.length

    thr_list = np.arange(thr_scale[0],thr_scale[1],thr_scale[2])
    acc_list = np.array([0 for x in thr_list])
    vis = visualizer()
    length = end-start
    acc_rec = ACC_record(acc_list,thr_list,length)
    acc_list = np.array([0.0 for x in thr_list])
    for i in tqdm(range(start,end)):
        #SegFileDataset process #unfair
        
        image , mask_orao =target_ds[i]
        mask_pre,CAMs = unet(image.unsqueeze(0).cuda())
        mask_pre = (1.0 - torch.softmax(mask_pre, dim=1))[:, 0]
        mask_pre = maskfilter(mask_pre.squeeze(),thr_list)
        IoU_list = msk2IoU_pre(mask_pre,mask_orao)
        acc_list += IoU_list

    print(acc_list/param.length)
    # print(acc_rec.acc_output()) 

def check_cls():
    param = TestParams()
    print('run test with parameter:')
    formate_output(param.__dict__)

    info_record = Info_Record('test')
    info_record.formate_record(param.__dict__,'both')
    init_seq = '{0} experment No: {1}  time : {2}'.format(\
            info_record.mode,info_record.id,info_record.time_sequence)
    print(init_seq)

    unet = eval(param.model)().cuda().eval()
    unet.load_state_dict(torch.load(param.model_weight, map_location='cpu'))
    unet.cuda().eval()

    test_dataset = TestDataset(param.val_root_dir,param.val_property_dir)
    start,end = 1000,2000
    acc_count = acc_counter()
    vis = visualizer()
    for i in tqdm(range(start,end)):
        image,cls_true = test_dataset[i]
        mask_pre,CAMs,cls_pre = unet(image.unsqueeze(0).cuda())
        mask_pre = (1.0 - torch.softmax(mask_pre, dim=1))[:, 0]
        cls_pre=torch.softmax(cls_pre,1)
        vis.save_image(image)
        print(cls_pre[0,cls_true])

        image_f = image*mask_pre.cpu()
        _,_,cls_f_pre  = unet(image_f.unsqueeze(0).cuda())
        cls_f_pre=torch.softmax(cls_f_pre,1)
        vis.save_image(image_f)
        print(cls_f_pre[0,cls_true])

        image_b = image*(1-mask_pre.cpu())
        _,_,cls_b_pre  = unet(image_b.unsqueeze(0).cuda())
        cls_b_pre=torch.softmax(cls_b_pre,1)
        vis.save_image(image_b)
        print(cls_b_pre[0,cls_true])

        ipdb.set_trace()
        #acc_count.count(cls_pre.argmax(1).cpu().numpy(),cls_true.cpu().numpy())

    avg = acc_count()
    print('cls_acc: {}'.format(avg))

def check_pm():
    param = TestParams()
    print('run test with parameter:')
    formate_output(param.__dict__)

    info_record = Info_Record('test')
    info_record.formate_record(param.__dict__,'both')
    init_seq = '{0} experment No: {1}  time : {2}'.format(\
            info_record.mode,info_record.id,info_record.time_sequence)
    print(init_seq)
    if 'scg' in param.model :
        unet = eval(param.model)(args=param).cuda().eval()
    else:
        unet = eval(param.model).model(args=param).cuda().eval()
    # unet.load_state_dict(torch.load(param.model_weight, map_location='cpu'))
    unet.cuda().eval()

    flops,params = get_model_complexity_info(unet,(3,224,224),as_strings=True,print_per_layer_stat=True)
    print(flops,params)

def check_param():
    param = TestParams()
    print('run test with parameter:')
    formate_output(param.__dict__)

    info_record = Info_Record('test')
    info_record.formate_record(param.__dict__,'both')
    init_seq = '{0} experment No: {1}  time : {2}'.format(\
            info_record.mode,info_record.id,info_record.time_sequence)
    print(init_seq)
    if 'scg' in param.model :
        unet = eval(param.model)(args=param).cuda().eval()
    else:
        # unet = eval(param.model).model(args=param).cuda().eval()
        unet = eval(param.model)().cuda().eval()
    # unet.load_state_dict(torch.load(param.model_weight, map_location='cpu'))
    unet.cuda().eval()

    flops,params = get_model_complexity_info(unet,(3,96,96),as_strings=True,print_per_layer_stat=True)
    print(flops,params)

def check_top1_top5():
    IoU_path = '/home/qxy/Desktop/BigGan/weight_result/IoU_data/CUB_SAL.json'
    cls_path = '/home/qxy/Desktop/BigGan/weight_result/cls_result/001791_CUB_5794'
    top1_path = os.path.join(cls_path,'classification1.json')
    top5_path = os.path.join(cls_path,'classification5.json')
    top1_acc = _ct_topk_acc(IoU_path,top1_path)
    top5_acc = _ct_topk_acc(IoU_path,top5_path)
    print('top1_acc: {}; top5_acc: {}'.format(top1_acc,top5_acc))

def _ct_topk_acc(IoU_path,topk_path):
    with open(IoU_path,'r') as f:
        IoU_lis = json.load(f)
    with open(topk_path,'r') as f:
        cls_lis = json.load(f)
    if len(IoU_lis) != len(cls_lis):
        raise Exception('length dont match')
    ct = np.array([0]*len(IoU_lis[0]))

    for i in range(len(IoU_lis)):
        iou_acc = (np.array(IoU_lis[i])>0.5).astype(int)
        cls_acc = int(cls_lis[i]>50)
        topk_acc = iou_acc*cls_acc
        ct+=topk_acc
    
    return ct/len(IoU_lis)

if __name__ == '__main__':
    main()
    # check()
    # check_visualization()
    # check_cls()
    # check_mask_thr()
    # check_ilsvrc_thr()
    # check_ilsvrc()
    # check_pm()
    # check_top1_top5()

# Threshold(model,thr=0.5,resize_to=128)