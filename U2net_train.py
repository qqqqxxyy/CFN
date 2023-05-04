#-*- coding: UTF-8 -*-
from __future__ import division
from copy import deepcopy
import os
import sys
import argparse
import json
import torch
import numpy as np
#from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, dataloader
import matplotlib
matplotlib.use("Agg")
import ipdb
#from utils.io_util import save_synthetic,save_generate
import json
from UNet.unet_model import UNet
from torchvision import transforms
from UNet.unet_cam import *
from UNet.u2net import *
from UNet.unet_ssol import *
from gan_mask_gen import MaskGenerator, MaskSynthesizing, it_mask_gen #通过gan网络生成mask
from data import OriDataset,ObjLocDataset,Trans,TransBbox
from data import CombData_pre as CombData,Random_noise
from metrics import ILSVRC_metrics, Localization, model_metrics, IoU, accuracy, F_max,ILSVRC_metrics
from BigGAN.gan_load import make_big_gan
from utils.postprocessing import connected_components_filter,\
    SegmentationInference, Threshold
from utils.utils import to_image, acc_counter
from utils.io_util import PATH_FILE, Train_Record,Pretrained_Read
from utils.prepared_util import list2json,load_path_file
from U2net_test import check_train,check_train_cls
PATH_FILE = load_path_file()

import time
DEFAULT_EVAL_KEY = 'id'
THR_EVAL_KEY = 'thr'
SEGMENTATION_RES = 128
BIGBIGAN_WEIGHTS = '../weight/pretrained_weights/BigBiGAN_x1.pth'
LATENT_DIRECTION = '../weight/pretrained_weights/bg_direction.pth'


MASK_SYNTHEZ_DICT = {
    'lighting': MaskSynthesizing.LIGHTING,
    'mean_thr': MaskSynthesizing.MEAN_THR,
}

class TrainParams(object):
    '''
    target:这是分割训练所需参数的容器
    容器接受任意外部传参，在保留外部传参的基础之上，
    还会添加__init__中所固定的故有参数
    '''
    def __init__(self, **kwargs):
        self.rate = 0.001
        self.steps_per_rate_decay = 7000
        self.rate_decay = 0.2
        self.n_steps = 12000

        self.latent_shift_r = 5.0
        self.batch_size = 95

        self.steps_per_log = 100
        self.steps_per_weight_save = 5000
        self.steps_per_validation = 4000
        self.test_samples_count = 1000
        self.model = 'UNet_cam'
        self.synthezing = MaskSynthesizing.LIGHTING
        self.mask_size_up = 0.5
        self.connected_components = True
        self.maxes_filter = True
        #evaluation 所需direction
        self.model_weight = None
        # self.mask_root_dir = 'CUB/segmentation'
        # self.image_root_dir = 'CUB/data'
        # self.image_property_dir = \
        # 'CUB/data_annotation/CUB_WSOL/train_list.json'
        self.syn_norm=True
        self.record_file=1
        self.val_property_dir = \
            'CUB/data_annotation/CUB_WSOL/val_list.json'
        # self.val_root_dir = None
        self.num_gpu = 1
        self.phase_lis = 1
        self.marker_lis = None
        self.dataset = 'CUB'
        self.noise_percent =None
        self.hdf5=False

        #可以从命令行读入的参数
        parser = argparse.ArgumentParser(description='GAN-based unsupervised segmentation train')
        parser.add_argument('--args', type=str, default=None, help='json with all arguments')
        parser.add_argument('--out', type=str)
        parser.add_argument('--gan_weights', type=str, default=BIGBIGAN_WEIGHTS)
        parser.add_argument('--bg_direction', type=str)
        #用于生成训练的显卡编号，本地只有一块卡因而写0
        parser.add_argument('--gen_devices', type=int, nargs='+', default=[0])
        parser.add_argument('--seed', type=int, default=2)
        parser.add_argument('--model', type=str, default=None)
        parser.add_argument('--z', type=str, default=None)
        parser.add_argument('--z_noise', type=float, default=0.0)
        parser.add_argument('--rate', type=float, default=None)
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--steps_per_validation', type=int, default=None)
        parser.add_argument('--n_steps', type=int, default=None)
        parser.add_argument('--steps_per_log', type=int, default=None)
        parser.add_argument('--model_weight', type=str, default=None)
        parser.add_argument('--num_gpu', type=int, default=None)
        parser.add_argument('--steps_per_weight_save', type=int, default=None)
        parser.add_argument('--syn_norm', type=bool, default=None)
        parser.add_argument('--record_file', type=int, default=None)
        parser.add_argument('--val_property_dir', type=str, default=None)
        parser.add_argument('--n_step_lis', type=int, nargs='+', default=None)
        parser.add_argument('--decay_lis', type=int, nargs='+', default=None)
        parser.add_argument('--batch_lis', type=int, nargs='+', default=None)
        parser.add_argument('--phase_lis', type=int, nargs='+', default=None)
        parser.add_argument('--marker_lis', type=str, nargs='+', default=None)
        # parser.add_argument('--image_property_dir', type=str, default=None)
        # parser.add_argument('--image_root_dir', type=str, default=None)
        # parser.add_argument('--val_root_dir', type=str, default=None)
        parser.add_argument('--dataset', type=str, default=None)
        parser.add_argument('--steps_per_rate_decay', type=int, default=None)
        parser.add_argument('--noise_percent', type=float, default=None)
        parser.add_argument('--hdf5', type=bool, default=None)

        args = parser.parse_args()
        self._add_parameter(args.__dict__)
        self._add_parameter(kwargs)

        self._complete_formed_paths('dataset_root','mask_root_dir')
        self._complete_formed_paths('dataset_root','image_root_dir')
        self._complete_formed_paths('dataset_root','val_root_dir')
        self._complete_formed_paths('dataset_root','image_property_dir')


        self._complete_paths('dataset_root','val_property_dir')


        if isinstance(self.synthezing, str):
            self.synthezing = MASK_SYNTHEZ_DICT[self.synthezing]
    
    def _add_parameter(self,dict_p):
        if type(dict_p).__name__ != 'dict':
            raise Exception('added parameter is not \'dict\' type')
        for key,val in dict_p.items():
            if val is not None:
                self.__dict__[key] = val

    def _complete_paths(self,rootpath,abspath):
        abss = self.__dict__[abspath]
        root_path = PATH_FILE[rootpath]
        self.__dict__[abspath] = os.path.join(root_path,abss)        
    
    def _complete_formed_paths(self,rootpath,abspath):
        root_path = PATH_FILE[rootpath]
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
    param = TrainParams()
    print('run train with parameter:')
    print(param.__dict__)
    info_record = Train_Record(file=param.record_file)
    info_record.record(__file__,'both')
    info_record.record(param.__dict__,'both')
    torch.random.manual_seed(param.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, param.gen_devices))

    init_seq = '{0} experment No: {1}  time : {2}'.format(\
            info_record.mode,info_record.id,info_record.time_sequence)
    print(init_seq)


    G = make_big_gan(param.gan_weights).eval().cpu()
    bg_direction = torch.load(param.bg_direction)
    mask_postprocessing = [connected_components_filter]
    params = [G,bg_direction,mask_postprocessing,param]
    devision_lis = ['synthetic', 'real', 'real']
    # devision_lis = ['synthetic', 'split', 'split']
    TrainOri = OriDataset(param.image_root_dir,param.image_property_dir,
                            hdf5=param.hdf5,
                         tsfm=transforms.Compose([transforms.Resize((156,156)), 
                                    transforms.RandomCrop(128), 
                                    # transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    Trans(['Normalize','standard']),
                                     ]))
    

    model = eval(param.model)().train().cuda()
    #parallelize, available for 1 to n gpus
    model = torch.nn.DataParallel(model.train().cuda(), range(param.num_gpu))
    pretrain_info = Pretrained_Read()

    if param.model_weight != None:
        pretrain_info.add_info_dict(param.model_weight)
        _transfer_load(model, pretrain_info.model_weight)

    # if param.model_weight != None:
    #     pretrain_info.add_info_dict(param.model_weight)
    #     try:
    #         model.module.load_state_dict(pretrain_info.model_weight)
    #     except :
    #         _model_load(model, pretrain_info.model_weight)
        
    # phase=1,2,3
    for i in param.phase_lis:
        param.n_steps = param.n_step_lis[i-1]
        param.steps_per_rate_decay = param.decay_lis[i-1]
        param.batch_size=param.batch_lis[i-1]
        devision = devision_lis[i-1]
        model.module.phase = i
        if param.marker_lis != None:
            model.module.initialize(param.marker_lis[i-1])
        CombLoader = CombData(TrainOri,MaskGenerator,devision,params)
        synthetic_score = train_segmentation(
            model,pretrain_info, CombLoader, param, info_record, param.gen_devices)

    info_record.record('train complete as plan!')
    print('Synthetic data score: {}'.format(synthetic_score))

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

    if list(model_dict.keys())[0].startswith('module.'):
            pretrained_dict = {'module.'+k: v for k, v in pretrained_dict.items()}

    print ("Weights cannot be loaded:")
    print ([k for k in model_dict.keys() if k not in pretrained_dict.keys()])
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)  

def _transfer_load(model, pretrained_dict):
    '''this function can help to load the consistent part and ignore the rest part
    if parameters in checkpoint not totally fit model (this happens when train CAM-based 
    model with imagenet pretrained backbone)
    '''
    model_dict = model.state_dict()
    pretrained_dict_b = {}
    for key in pretrained_dict.keys():
        str_head = '.'
        key_b = key.split('.')
        if 'A_outc' in key_b[0]:
            key_bb = 'B_'+key_b[0].split('A_')[1]+'.'+str_head.join(key_b[1:])
            pretrained_dict_b.update({key_bb:pretrained_dict[key]}) 
        else:
            key_b = key_b[0].split('_A')[0]+'_B.'+str_head.join(key_b[1:])
            pretrained_dict_b.update({key_b:pretrained_dict[key]}) 
            # print(key_b)
            # ipdb.set_trace()
    pretrained_dict.update(pretrained_dict_b)
    if list(model_dict.keys())[0].startswith('module.'):
            pretrained_dict = {'module.'+k: v for k, v in pretrained_dict.items()}
    print ("Weights cannot be loaded:")
    print ([k for k in model_dict.keys() if k not in pretrained_dict.keys()])
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)  

def read_train_np(paths):
    '''加载z向量和每一个z向量所对应的类别
    '''
    zs_path = paths
    zs_cls_path = paths.split('.npy')[0]+'_cls.npy'
    zs = torch.from_numpy(np.load(zs_path))
    zs_cls = torch.from_numpy(np.load(zs_cls_path))
    return zs, zs_cls

def train_segmentation(model,pretrain_info, CombLoader,params,info_record,gen_devices):
    '''target:训练model
    input:
    G: BigGAN生成模型
    bg_direction: 纺射矩阵
    mdoel: 分割模型
    params: 训练参数
    gen_devices: gpu数量
    '''
    model.train()
    params.batch_size = params.batch_size // len(gen_devices)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params.rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, params.steps_per_rate_decay, params.rate_decay)
    if pretrain_info.optimizer_load==True:
        optimizer.load_state_dict(pretrain_info.optimizer_weight)
        lr_scheduler.load_state_dict(pretrain_info.scheduler_weight)
        print('successfule load')

    # model.module.initialize_weights('B')
    model.module.loss_init()
    index = 0
    # model.module.geo_rot_initial()
    
    while True:
        #加载权重and预处理

        image,clas,mask = CombLoader[index]
        index += 1
        model.zero_grad()

        logits = model(image)
        if 'cls' in params.model:
            '''如果对于分类标签的模型
            '''
            loss = model.module.get_loss(logits,clas)
            #统计准确率
            model.module.loss_count(loss,logits,clas)
        elif 'fb' in params.model:
            '''对于弱监督模型
            '''
            loss = model.module.get_loss(logits,clas,mask)
            cls_pre = logits[0].detach()
            model.module.loss_count(loss,cls_pre,clas)
        elif 'aug' in params.model:
            '''anti-noise 模块'''
            model.module.geo_rot_special()
            image_T = model.module.image_rotation(image)

            mask_A,mask_B = model(image_T)

            mask_A_T,mask_B_T = model.module.image_revise_rotation(mask_A,mask_B)

            logtis_t = (mask_A_T,mask_B_T)
            loss = model.module.get_loss(logits,logtis_t,clas.shape[0],mask)

            model.module.loss_count(loss)
        else:
            if params.noise_percent != None:
                mask = Random_noise(mask,params.noise_percent)
                mask = (mask>0.5).long()
            loss = model.module.get_loss(logits,clas.shape[0],mask)
            model.module.loss_count(loss)
        

        #优化

        loss[0].backward()
        optimizer.step()
        lr_scheduler.step()
        # t2 = time.time()
        # print(t1-t0,' ',t2-t1)
        #每 steps_per_log 进行记录一次
        if index% params.steps_per_log == 0 and index>0:
            
            loss_ct = model.module.loss_output()
            output = '{}% | step {}: loss:{}'\
                .format(int(100.0 * index / params.n_steps), index, loss_ct)
            print(output)
            info_record.record(output,'both')
            model.module.loss_reset()

        #每steps_per_weight_save保存权重一次
        if index% params.steps_per_weight_save == 0 and index>0:
            # info_record.save_result_weight(model.module.state_dict(),'{}_{}'.format(index,model.module.phase))   
            info_record.save_checkpoint(model.module,optimizer, \
            lr_scheduler,index,'{}_{}'.format(model.module.phase,index))         


        #每steps_per_validation 用验证集检测一次
        if index%params.steps_per_validation == 0 or index >= params.n_steps:# and index>0:
            model.module.eval()
            tsfm=transforms.Compose([ transforms.Resize((128,128)),
                transforms.ToTensor(), Trans(['Normalize','standard']) ])
            tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(128,128)])])

            val_ds = ObjLocDataset(params.val_root_dir,params.val_property_dir,tsfm,tsfm_bbox,params.hdf5)
            val_dl = torch.utils.data.DataLoader(val_ds,1,shuffle=False)
            if 'cls' in params.model:
                # ipdb.set_trace()
                info = check_train_cls(model,val_dl)
                info_record.record(info,'log_detail')
                print('class_acc:',info)
            else:
                info = check_train(model,val_dl)
                info_record.record(info,'log_detail')
                print('0.5:',info['0.5'],';0.9:',info['0.9'],';mIoU:',info['miou'],';box_v2:',info['box_v2'])
            model.module.train()

        if index >= params.n_steps:
            break
        

    model.module.eval()
    info_record.save_result_weight(model.module.state_dict(),'_{}'.format(model.module.phase))
    return 0

if __name__ == '__main__':
    main()