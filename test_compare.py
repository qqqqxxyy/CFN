#-*- coding: UTF-8 -*-
from random import randint
from re import I, U
from UNet.unet_model import UNet
from data import TestDataset
#SegmentationInference, Threshold,resize_min_edge
from metrics import F_max, IoU, accuracy\
                , precision_recall,croped_resize
from utils.io_util import ROOT_PATH, formate_output
import os
import sys
import argparse
import json
import torch
import cv2
from torchvision.transforms import ToPILImage, ToTensor, Resize
import numpy as np
from gan_mask_gen import MaskSynthesizing
#from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import ipdb
import torch.nn.functional as F
from utils.io_util import Info_Record
from UNet.unet_cam import *
from compare_model import Spa_vgg,CAM_vgg,I2c_vgg,ACoL_vgg
import UNet.SGL_ResNet_adaptive as SGL_ResNet_adaptive
from gan_mask_gen import MaskGenerator, MaskSynthesizing, it_mask_gen #通过gan网络生成mask
from data import SegFileDataset,TrainDataset,CombData
from metrics import *
#model_metrics, IoU, accuracy, F_max,Localization,mask2IoU,cal_IoU
from BigGAN.gan_load import make_big_gan
from utils.postprocessing import connected_components_filter,\
    SegmentationInference, Threshold,CAMInference
from utils.utils import to_image,IoU_Manager
from utils.visual_util import visualize_tensor,visualizer
from tqdm import tqdm
from ptflops import get_model_complexity_info
PATHS_FILE='../weight_result/paths.json'
UNET_PATH = '/home/qxy/Desktop/BigGan/weight_result/results/CUB/segmentation.pth'
class TestParams(object):

    def __init__(self):
        #固定参数
        self.batch_size= 1
        self.mask_root_dir = 'CUB/segmentation'
        self.image_root_dir = 'CUB/data'
        self.image_property_dir = \
            'CUB/data_annotation/CUB_WSOL/test_list.json'
            # '/home/qxy/Desktop/datasets/CUB/data_annotation/CUB_WSOL/test_list.json'
            
        self.bg_direction = '../weight_result/weights/bg_direction.pth'
        self.gan_weights='../weight_result/weights/BigBiGAN_x1.pth'
        self.latent_shift_r = 5.0
        self.model = 'UNet_cam'
        self.mask_size_up = 0.5
        self.save_dir = None
        self.synthezing = MaskSynthesizing.LIGHTING
        self.maxes_filter = True
        self.syn_norm=False
        #可以从命令行读入的参数
        parser = argparse.ArgumentParser(description='GAN-based unsupervised segmentation test')
        parser.add_argument('--unet_load_paths', type=str, default=UNET_PATH)
        parser.add_argument('--model', type=str, default=None)
        parser.add_argument('--image_property_dir', type=str, default=None)
        parser.add_argument('--image_root_dir', type=str, default=None)
        parser.add_argument('--save_dir', type=str, default=None)
        parser.add_argument('--pre_thr', type=float, default=None)
        parser.add_argument('--mode', type=str, default=None)
        parser.add_argument('--z', type=str, default=None)
        parser.add_argument('--z_noise', type=float, default=0.2)
        parser.add_argument('--length', type=int, default=500)
        # parser.add_argument('--loss_local_factor', type=float, default=0.008)
        # parser.add_argument('--local_seed_num', type=int, default=3)
        # parser.add_argument('--loss_global_factor', type=float, default=0.001)
        # parser.add_argument('--scg_fosc_th', type=float, default=0.1)
        # parser.add_argument('--scg_sosc_th', type=float, default=0.5)
        # parser.add_argument('--scg_order', type=int, default=2)
        # parser.add_argument('--scg_so_weight', type=float, default=2)
        # parser.add_argument('--scg_fg_th', type=float, default=0.1)
        # parser.add_argument('--scg_bg_th', type=float, default=0.05)
        parser.add_argument('--scg_blocks', type=str, default='45')
        parser.add_argument('--scg', type=bool, default=True)
        parser.add_argument('--scg_fosc_th', type=float, default=0.15)
        parser.add_argument('--scg_sosc_th', type=float, default=0.5)
        parser.add_argument('--scg_order', type=int, default=2)
        parser.add_argument('--scg_so_weight', type=float, default=0.1)
        parser.add_argument('--scg_fg_th', type=float, default=0.13)
        parser.add_argument('--scg_bg_th', type=float, default=0.17)

        args = parser.parse_args()       

        for key, val in args.__dict__.items():
            if val is not None:
                ##__dict__中存放self.XXX这样的属性，作为键和值存储
                #下面代码功能等同于将dict中所有元素作self.key = val的相应操作
                self.__dict__[key] = val 
        self._complete_paths('dataset_root','mask_root_dir')
        self._complete_paths('dataset_root','image_root_dir')
        self._complete_paths('dataset_root','image_property_dir')

    def _complete_paths(self,rootpath,abspath):
        abss = self.__dict__[abspath]
        with open(PATHS_FILE) as f:
            path_dict = json.load(f)
            root_path = path_dict[rootpath]
        self.__dict__[abspath] = os.path.join(root_path,abss)   

class TestParams_jupyter(object):

    def __init__(self):
        #固定参数
        self.batch_size= 1
        self.mask_root_dir = '/home/qxy/Desktop/datasets/CUB/segmentation'
        self.image_root_dir = '/home/qxy/Desktop/datasets/CUB/data'
        self.image_property_dir = \
            '/home/qxy/Desktop/datasets/CUB/data_annotation/CUB_WSOL/test_list.json'

        #可以从命令行读入的参数
        parser = argparse.ArgumentParser(description='GAN-based unsupervised segmentation test')
        parser.add_argument('--unet_load_paths', type=str, default=UNET_PATH)
        parser.add_argument('--n_steps', type=int, default=None)
        parser.add_argument('--save_dir', type=str, default=None)

        args = parser.parse_args([])       

        for key, val in args.__dict__.items():
            if val is not None:
                ##__dict__中存放self.XXX这样的属性，作为键和值存储
                #下面代码功能等同于将dict中所有元素作self.key = val的相应操作
                self.__dict__[key] = val 

def _model_load(model, pretrained_dict):
    '''if parameters in checkpoint not totally fit model (this happens when train CAM-based 
    model with imagenet pretrained backbone), this function can help to load the consistent
    part and ignore the rest part
    '''
    model_dict = model.state_dict()
    # model_dict_keys = [v.replace('module.', '') for v in model_dict.keys() if v.startswith('module.')]
    if list(model_dict.keys())[0].startswith('module.'):
            pretrained_dict = {'module.'+k: v for k, v in pretrained_dict.items()}

    if list(model_dict.keys())[0].startswith('conv'):
        pre2local_keymap = [('features.{}.weight'.format(i), 'conv1_2.{}.weight'.format(i)) for i in range(10)]
        pre2local_keymap += [('features.{}.bias'.format(i), 'conv1_2.{}.bias'.format(i)) for i in range(10)]
        pre2local_keymap += [('features.{}.weight'.format(i + 10), 'conv3.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 10), 'conv3.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.weight'.format(i + 17), 'conv4.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 17), 'conv4.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.weight'.format(i + 24), 'conv5.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('features.{}.bias'.format(i + 24), 'conv5.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap = dict(pre2local_keymap)
        pretrained_dict = {pre2local_keymap[k] if k in pre2local_keymap.keys() else k: v for k, v in
                           pretrained_dict.items()}


    # print pretrained_dict.keys()
    # print model.state_dict().keys()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    print ("Weights cannot be loaded:")
    print ([k for k in model_dict.keys() if k not in pretrained_dict.keys()])

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)  
def main():

    param = TestParams()
    print('run test with parameter:')
    formate_output(param.__dict__)

    info_record = Info_Record('test')
    info_record.formate_record(param.__dict__,'both')

    init_seq = '{0} experment No: {1}  time : {2}'.format(\
            info_record.mode,info_record.id,info_record.time_sequence)
    print(init_seq)

    unet = eval(param.model)().cuda().eval()
    unet.load_state_dict(torch.load(param.unet_load_paths, map_location='cpu'))
    unet.cuda().eval()

    target_ds = SegFileDataset(param,crop=True,size=(128,128))
    segmentation_dl = torch.utils.data.DataLoader(target_ds,param.batch_size,shuffle=False)
    
    outdict = model_metrics(Threshold(unet, thr=0.5, resize_to=128), segmentation_dl, \
                                stats=(IoU, accuracy,Localization), n_steps = param.n_steps)

    output = 'Thresholded:\n{}'.format( outdict )
    print(output)
    info_record.record(output)
    info_record.record('test complete as plan!')

def check():
    param = TestParams()
    print('run test with parameter:')
    formate_output(param.__dict__)

    info_record = Info_Record('test')
    info_record.formate_record(param.__dict__,'both')
    if 'UNet' in param.model:
            unet = eval(param.model)().cuda().eval()
            ckpt = torch.load(param.unet_load_paths, map_location='cpu')
            unet.load_state_dict(ckpt)
            unet = torch.nn.DataParallel(unet.train().cuda(), [0])
    else:
        if 'CUB'in param.image_root_dir:
            unet = eval(param.model).model(num_classes=200,args=param).train().cuda()
        else:
            unet = eval(param.model).model(num_classes=1000,args=param).train().cuda()
        unet = torch.nn.DataParallel(unet.train().cuda(), [0])
        ckpt = torch.load(param.unet_load_paths, map_location='cpu')
        unet.load_state_dict(ckpt)
    unet.cuda().eval()
    vis = visualizer()

    G = make_big_gan(param.gan_weights).eval().cpu()
    bg_direction = torch.load(param.bg_direction)
    mask_postprocessing = [connected_components_filter]
    params = [G,bg_direction,mask_postprocessing,param]
    CombLoader = CombData(TrainDataset,MaskGenerator,'real',params)
    
    
    for t in tqdm(range(0,param.length,1)):
        # image,mask = target_ds[t]
        # image2,clas,mask = CombLoader[t]
        # tar_ds2 = SegFileDataset(param,crop=False,size=(224,224))
        # image2, mask_orao = tar_ds2[t]
        
        tar_ds2 = SegFileDataset(param,crop=False,size=(224,224))
        image2, mask_orao = tar_ds2[t]
        image2 = image2.unsqueeze(0)
        '''
        #通过croped image 得到 croped mask
        ftmp_list,_ = unet(image2.cuda()) 
        featmp,ftmp_fo,ftmp_bc=ftmp_list
        featmp = torch.mean(featmp,dim=1)
        ftmp_fo = torch.mean(ftmp_fo,dim=1)
        ftmp_bc = torch.mean(ftmp_bc,dim=1)
        featmp = F.interpolate(featmp.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=True)
        ftmp_fo = F.interpolate(ftmp_fo.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=True)
        ftmp_bc = F.interpolate(ftmp_bc.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=True)
        '''
        mask_pre,_ = unet(image2.cuda())
        if 'UNet' in param.model:
            mask_pre = (1.0 - torch.softmax(mask_pre, dim=1))[:, 0]
        else:
            mask_pre = ivr_p(mask_pre,0)

        mask_pre = F.interpolate(mask_pre.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=True)
        _,cor_real,cor_pre = mask2IoU(mask_orao,mask_pre.squeeze(),pre_thr=param.pre_thr)
        save_paths = '/home/qxy/Desktop/BigGan/weight_result/results/picture_casual/'
        if param.save_dir != 'None' and param.save_dir != None:
            save_paths = os.path.join(save_paths,param.save_dir)
            os.makedirs(save_paths,exist_ok=True)
        
        if len(image2.shape)==4:
            image2 =  image2.squeeze(0)
        if len(mask_pre.she)==4:
            mask_pre =  mask_pre.squeeze(0)
        ipdb.set_trace()
        vis.save_htmp(mask_pre,image= image2,paths=[save_paths,'{}_0_{}.png'.format(t,param.mode)])
        vis.save_htmp(mask_pre,image= image2,bbox_pre = cor_pre[0],paths=[save_paths,'{}_1_{}.png'.format(t,param.mode)])
        vis.save_htmp(mask_pre,image= image2,bbox_pre = cor_pre[0],\
            bbox_gt = cor_real[0],paths=[save_paths,'{}_2_{}.png'.format(t,param.mode)])
        
        
        '''
        img_f = image2.cpu().detach()*mask_pre.cpu().detach()
        img_b = image2.cpu().detach()*(1-mask_pre.cpu().detach())
        # vis.save_htmp(mask_pre)
        # vis.save_htmp(CAMs)
        # vis.save_image(image,mask=mask_pre.squeeze())
        _,cor_real,cor_pre = mask2IoU(mask_orao,mask_pre.squeeze(),pre_thr=param.pre_thr)

        cor_pre = croped_resize(cor_pre,image2)
        iou = cor2IoU(cor_real,cor_pre[0])
        mask = mask.float()
        '''
        '''
        # ipdb.set_trace()
        image2 = F.interpolate(image2,size=(224,224),mode='bilinear',align_corners=True)
        mask = F.interpolate(mask.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=True).squeeze(0)
        mask = mask>0.5
        '''
        '''
        # print(iou)
        save_paths = '/home/qxy/Desktop/BigGan/weight_result/results/picture_casual/'
        if param.save_dir != 'None' and param.save_dir != None:
            save_paths = os.path.join(save_paths,param.save_dir)
            os.makedirs(save_paths,exist_ok=True)
        vis.save_htmp(featmp,paths=[save_paths,'{}_0_{}.png'.format(t,param.mode)])
        vis.save_htmp(ftmp_fo,paths=[save_paths,'{}_1_{}.png'.format(t,param.mode)])
        vis.save_htmp(ftmp_bc,paths=[save_paths,'{}_2_{}.png'.format(t,param.mode)])
        # vis.save_image(image2,paths=[save_paths],bbox = cor_pre[0])
        # ipdb.set_trace()
        # if iou[0]>0.9:
        # vis.save_htmp(mask_pre,image= image2,paths=[save_paths,'{}_0_{}.png'.format(t,param.mode)])
        # vis.save_htmp(mask_pre,image= image2,bbox_pre = cor_pre[0],paths=[save_paths,'{}_1_{}.png'.format(t,param.mode)])
        # vis.save_htmp(mask_pre,image= image2,bbox_pre = cor_pre[0],\
        #     bbox_gt = cor_real[0],paths=[save_paths,'{}_2_{}.png'.format(t,param.mode)])
        '''
        '''
        vis.save_graymp(mask_pre,paths=[save_paths,'{}_0_{}.png'.format(t,param.mode)])
        vis.save_image(img_f,paths=[save_paths,'{}_1_{}.png'.format(t,param.mode)])
        vis.save_image(img_b,paths=[save_paths,'{}_2_{}.png'.format(t,param.mode)])
        
        vis.save_image(image2,paths=[save_paths,'{}_0_{}.png'.format(t,param.mode)])
        vis.save_image(image2,mask = mask,paths=[save_paths,'{}_1_{}.png'.format(t,param.mode)])
        vis.save_htmp(mask_pre,paths=[save_paths,'{}_2_{}.png'.format(t,param.mode)])
        '''

def check_wrong():
    param = TestParams()
    print('run test with parameter:')
    formate_output(param.__dict__)

    info_record = Info_Record('test')
    info_record.formate_record(param.__dict__,'both')

    unet = eval(param.model)().cuda().eval()
    unet.load_state_dict(torch.load(param.unet_load_paths, map_location='cpu'))
    unet.cuda().eval()

    target_ds = SegFileDataset(param,crop=True,size=(128,128))
    mask_predictor = Threshold(unet,thr=0.5)
    vis = visualizer()
    mod = param.unet_load_paths.split('/')[-1].split('.')[0]
    ct=0

    for i in tqdm(range(1000,2000)):
        image,mask = target_ds[i]
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

def check_thr():
    param = TestParams()
    print('run test with parameter:')
    formate_output(param.__dict__)

    info_record = Info_Record('test')
    info_record.formate_record(param.__dict__,'both')
    init_seq = '{0} experment No: {1}  time : {2}'.format(\
            info_record.mode,info_record.id,info_record.time_sequence)
    print(init_seq)

    if 'CUB'in param.image_root_dir:
        unet = eval(param.model).model(num_classes=200,args=param).train().cuda()
    else:
        unet = eval(param.model).model(num_classes=1000,args=param).train().cuda()
    # ipdb.set_trace()
    unet = torch.nn.DataParallel(unet.train().cuda(), [0])
    unet.cuda().eval()
    if param.unet_load_paths != None:
        ckpt = torch.load(param.unet_load_paths, map_location='cpu')
        try:
            unet.load_state_dict(ckpt)
        except :
            _model_load(unet, ckpt)


    target_ds = SegFileDataset(param,crop=True,size=(224,224))
    # tar_ds2 = SegFileDataset(param,crop=False,size=(224,224))

    thr_scale = [0.01,0.30,0.01]
    start,end = 0,5794

    thr_list = np.arange(thr_scale[0],thr_scale[1],thr_scale[2])
    acc_list = np.array([0 for x in thr_list])
    vis = visualizer()
    length = end-start
    acc_rec = ACC_record(acc_list,thr_list,length)

    for i in tqdm(range(start,end)):
        
        image,mask_orao = target_ds[i]
        mask_pre,CAMs = unet(image.unsqueeze(0).cuda())
        
        # mask_pre = ivr_p(mask_pre,0)
        # mask_pre = F.interpolate(mask_pre.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=True)
        # CAMs = -CAMs
        # CAMs = (CAMs-CAMs.min())/(CAMs.max()-CAMs.min())
        mask_pre = (mask_pre-mask_pre.min())/(mask_pre.max()-mask_pre.min())
        mask_pre = F.interpolate(mask_pre.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=True)
        # mask_pre = (1.0 - torch.softmax(mask_pre, dim=1))[:, 0]
        #fair crop
        # 得到原图片的mask
        # image_orao, mask_orao = tar_ds2[i]
        cor_orao = mask2cor(mask_orao,0.5)
        # 得到预测的mask list并转化为原图片尺寸
        cor_gen = mask2cor(mask_pre.squeeze(),thr_list)
        # cor_gen = croped_resize(cor_gen,image)
        # ipdb.set_trace()
        # 得到iou list并由此得到acc list
        IoU_list = cor2IoU(cor_orao,cor_gen)

        # # unfair crop
        # cor_true = mask2cor(mask,0.5)
        # cor_gen = mask2cor(mask_pre.squeeze(),thr_list)
        # IoU_list = cor2IoU(cor_true,cor_gen)

        acc_lis = IoU2acc(IoU_list)       
        acc_list += acc_lis
        acc_rec.count(IoU_list)

    print(acc_rec.acc_output()) 
    acc_list = acc_list/length
    print(acc_list)
    # info_record.record(str(acc_list))

def check_mask_thr():
    param = TestParams()
    print('run test with parameter:')
    formate_output(param.__dict__)

    info_record = Info_Record('test')
    info_record.formate_record(param.__dict__,'both')
    init_seq = '{0} experment No: {1}  time : {2}'.format(\
            info_record.mode,info_record.id,info_record.time_sequence)
    print(init_seq)


    if 'CUB'in param.image_root_dir:
        unet = eval(param.model).model(num_classes=200,args=param).train().cuda()
    else:
        unet = eval(param.model).model(num_classes=1000,args=param).train().cuda()
    # ipdb.set_trace()
    unet = torch.nn.DataParallel(unet.train().cuda(), [0])
    unet.cuda().eval()
    if param.unet_load_paths != None:
        ckpt = torch.load(param.unet_load_paths, map_location='cpu')
        try:
            unet.load_state_dict(ckpt)
        except :
            _model_load(unet, ckpt)

    target_ds = SegFileDataset(param,crop=True,size=(224,224))
    tar_ds2 = SegFileDataset(param,crop=False,size=128)
    val_ds = TestDataset(param.image_root_dir,param.image_property_dir)

    thr_scale = [0.20,0.40,0.01]
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
        
        mask_pre = (mask_pre-mask_pre.min())/(mask_pre.max()-mask_pre.min())
        mask_pre = F.interpolate(mask_pre.unsqueeze(0),size=(128,128),mode='bilinear',align_corners=True).squeeze()
        mask_orao = F.interpolate(mask_orao.unsqueeze(0).unsqueeze(0).float(),size=(128,128),mode='bilinear',align_corners=True).squeeze().bool()

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
    unet.load_state_dict(torch.load(param.unet_load_paths, map_location='cpu'))
    unet.cuda().eval()

    test_dataset = TestDataset(param.image_root_dir,param.image_property_dir)
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
        #acc_count.count(cls_pre.argmax(1).cpu().numpy(),cls_true.cpu().numpy())

    avg = acc_count()
    print('cls_acc: {}'.format(avg))

def check_ilsvrc_thr():
    param = TestParams()
    print('run test with parameter:')
    formate_output(param.__dict__)

    info_record = Info_Record('test')
    info_record.formate_record(param.__dict__,'both')
    init_seq = '{0} experment No: {1}  time : {2}'.format(\
            info_record.mode,info_record.id,info_record.time_sequence)
    print(init_seq)
    if 'CUB'in param.image_root_dir:
        unet = eval(param.model).model(num_classes=200,args=param).train().cuda()
    else:
        unet = eval(param.model).model(num_classes=1000,args=param).train().cuda()
    unet = torch.nn.DataParallel(unet.train().cuda(), [0])
    ckpt = torch.load(param.unet_load_paths, map_location='cpu')
    unet.load_state_dict(ckpt)
    unet.cuda().eval()

    val_ds = TestDataset(param.image_root_dir,param.image_property_dir,False)
    # val_dl = torch.utils.data.DataLoader(val_ds,1,shuffle=False)

    thr_scale = [0.10,0.30,0.01]
    start,end = 0,50000

    thr_list = np.arange(thr_scale[0],thr_scale[1],thr_scale[2])
    acc_list = np.array([0 for x in thr_list])
    length = end-start
    acc_rec = ACC_record(acc_list,thr_list,length)
    iou_men = IoU_Manager()
    for i in tqdm(range(start,end)):
        image,_ ,bbox = val_ds[i]
        
        mask_pre,_ = unet(image.unsqueeze(0).cuda())
        mask_pre = ivr_p(mask_pre,0)
        mask_pre = F.interpolate(mask_pre.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=True)
        
        # mask_pre = Threshold(unet,thr=0.5,resize_to=128)(image.unsqueeze(0))
        
        
        #generate bbox from htmp
        cor_gen = mask2cor(mask_pre.squeeze(),thr_list)
        #generate bbox from mask
        # cor_gen  = mask2cor(mask_pre.squeeze(),thr_list)
        #cor_gen = croped_resize(cor_gen,size)
        IoU_list = cor2IoU(bbox,cor_gen)
        iou_men.add(IoU_list)
        acc_rec.count(IoU_list)
        # acc_lis = IoU2acc(IoU_list)
        # acc_list+=acc_lis

    print(acc_rec.acc_output())    
    iou_men.save('/home/qxy/Desktop/BigGan/weight_result/IoU_data/ILSVRC.json')
    # acc_list = acc_list/length
    # print(acc_list)
    # info_record.record(str(acc_list))

def check_ilsvrc():
    param = TestParams()
    print('run test with parameter:')
    formate_output(param.__dict__)

    info_record = Info_Record('test')
    info_record.formate_record(param.__dict__,'both')
    if 'UNet' in param.model:
            unet = eval(param.model)().cuda().eval()
            ckpt = torch.load(param.unet_load_paths, map_location='cpu')
            unet.load_state_dict(ckpt)
            unet = torch.nn.DataParallel(unet.train().cuda(), [0])
    else:
        if 'CUB'in param.image_root_dir:
            unet = eval(param.model).model(num_classes=200,args=param).train().cuda()
        else:
            unet = eval(param.model).model(num_classes=1000,args=param).train().cuda()
        unet = torch.nn.DataParallel(unet.train().cuda(), [0])
        ckpt = torch.load(param.unet_load_paths, map_location='cpu')
        unet.load_state_dict(ckpt)
    unet.cuda().eval()
    vis = visualizer()

    G = make_big_gan(param.gan_weights).eval().cpu()
    bg_direction = torch.load(param.bg_direction)
    mask_postprocessing = [connected_components_filter]
    params = [G,bg_direction,mask_postprocessing,param]
    CombLoader = CombData(TrainDataset,MaskGenerator,'real',params)
    
    
    for t in tqdm(range(0,param.length,1)):
        # image,mask = target_ds[t]
        # image2,clas,mask = CombLoader[t]
        # tar_ds2 = SegFileDataset(param,crop=False,size=(224,224))
        # image2, mask_orao = tar_ds2[t]
        
        val_ds = TestDataset(param.image_root_dir,param.image_property_dir)
        image2, _,cor_real = val_ds[t]
        cor_real = [[int(x) for x in cor_real[0]]]
        image2 = image2.unsqueeze(0)
        '''
        #通过croped image 得到 croped mask
        ftmp_list,_ = unet(image2.cuda()) 
        featmp,ftmp_fo,ftmp_bc=ftmp_list
        featmp = torch.mean(featmp,dim=1)
        ftmp_fo = torch.mean(ftmp_fo,dim=1)
        ftmp_bc = torch.mean(ftmp_bc,dim=1)
        featmp = F.interpolate(featmp.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=True)
        ftmp_fo = F.interpolate(ftmp_fo.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=True)
        ftmp_bc = F.interpolate(ftmp_bc.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=True)
        '''
        mask_pre,_ = unet(image2.cuda())
        if 'UNet' in param.model:
            mask_pre = (1.0 - torch.softmax(mask_pre, dim=1))[:, 0]
        else:
            mask_pre = ivr_p(mask_pre,0)

        mask_pre = F.interpolate(mask_pre.unsqueeze(0),size=(224,224),mode='bilinear',align_corners=True)
        _,_,cor_pre = mask2IoU(mask_pre.squeeze(),mask_pre.squeeze(),pre_thr=param.pre_thr)
        save_paths = '/home/qxy/Desktop/BigGan/weight_result/results/picture_casual/'
        if param.save_dir != 'None' and param.save_dir != None:
            save_paths = os.path.join(save_paths,param.save_dir)
            os.makedirs(save_paths,exist_ok=True)
        
        if len(image2.shape)==4:
            image2 =  image2.squeeze(0)
        if len(mask_pre.shape)==4:
            mask_pre =  mask_pre.squeeze(0)
        
        vis.save_htmp(mask_pre,image= image2,paths=[save_paths,'{}_0_{}.png'.format(t,param.mode)])
        vis.save_htmp(mask_pre,image= image2,bbox_pre = cor_pre[0],paths=[save_paths,'{}_1_{}.png'.format(t,param.mode)])
        vis.save_htmp(mask_pre,image= image2,bbox_pre = cor_pre[0],\
            bbox_gt = cor_real[0],paths=[save_paths,'{}_2_{}.png'.format(t,param.mode)])
        
        '''
        #save original image
        vis.save_image(image2,paths=[save_paths,'{}_0_{}.png'.format(t,param.mode)])
        '''

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
    # unet.load_state_dict(torch.load(param.unet_load_paths, map_location='cpu'))
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
        unet = eval(param.model).model(args=param).cuda().eval()
        # unet = eval(param.model)().cuda().eval()
    # unet.load_state_dict(torch.load(param.unet_load_paths, map_location='cpu'))
    unet.cuda().eval()

    flops,params = get_model_complexity_info(unet,(3,224,224),as_strings=True,print_per_layer_stat=True)
    print(flops,params)

def _model_load(model, pretrained_dict):
    '''this function can help to load the consistent part and ignore the rest part
    if parameters in checkpoint not totally fit model (this happens when train CAM-based 
    model with imagenet pretrained backbone)
    '''
    model_dict = model.state_dict()
    # model_dict_keys = [v.replace('module.', '') for v in model_dict.keys() if v.startswith('module.')]
    if list(model_dict.keys())[0].startswith('module.'):
            pretrained_dict = {'module.'+k: v for k, v in pretrained_dict.items()}
    if list(model_dict.keys())[0].startswith('module.conv'):
        pre2local_keymap = [('module.features.{}.weight'.format(i), 'module.conv1_2.{}.weight'.format(i)) for i in range(10)]
        pre2local_keymap += [('module.features.{}.bias'.format(i), 'module.conv1_2.{}.bias'.format(i)) for i in range(10)]
        pre2local_keymap += [('module.features.{}.weight'.format(i + 10), 'module.conv3.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('module.features.{}.bias'.format(i + 10), 'module.conv3.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap += [('module.features.{}.weight'.format(i + 17), 'module.conv4.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('module.features.{}.bias'.format(i + 17), 'module.conv4.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap += [('module.features.{}.weight'.format(i + 24), 'module.conv5.{}.weight'.format(i)) for i in range(7)]
        pre2local_keymap += [('module.features.{}.bias'.format(i + 24), 'module.conv5.{}.bias'.format(i)) for i in range(7)]
        pre2local_keymap = dict(pre2local_keymap)
        pretrained_dict = {pre2local_keymap[k] if k in pre2local_keymap.keys() else k: v for k, v in
                           pretrained_dict.items()}


    # print pretrained_dict.keys()
    # print model.state_dict().keys()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    print ("Weights cannot be loaded:")
    print ([k for k in model_dict.keys() if k not in pretrained_dict.keys()])
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)  


if __name__ == '__main__':
    # main()
    '''cub thr'''
    # check_thr()
    # check_mask_thr()
    check()
    # check_cls()
    '''ilsvrc thr'''
    # check_ilsvrc_thr()
    # check_ilsvrc()
    # check_param()

# Threshold(model,thr=0.5,resize_to=128)