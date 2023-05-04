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
from compare_model import Spa_vgg,CAM_vgg,I2c_vgg,ACoL_vgg
from compare_model.inceptionv3 import inceptionv3_ACoL,inceptionv3_CAM,inceptionv3_SPA
from gan_mask_gen import MaskGenerator, MaskSynthesizing, it_mask_gen #通过gan网络生成mask
from data import CombData_pre as CombData,Random_noise,OriDataset,ObjLocDataset,Trans,TransBbox,SegFileDataset
from metrics import ILSVRC_metrics, Localization, model_metrics, IoU, accuracy, F_max,ILSVRC_metrics
from BigGAN.gan_load import make_big_gan
from utils.postprocessing import connected_components_filter,\
    SegmentationInference, Threshold
from utils.utils import to_image, acc_counter
from utils.io_util import PATH_FILE, Train_Record,Pretrained_Read
from utils.prepared_util import list2json,load_path_file
from U2net_test import check_train,check_train_cls
from compare_model import my_optim
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
        self.num_gpu=1
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
        self.syn_norm=False
        self.record_file=1
        self.val_property_dir = \
            'CUB/data_annotation/CUB_WSOL/val_list.json'
        self.dataset = 'CUB'
        self.hdf5=False

        #可以从命令行读入的参数
        parser = argparse.ArgumentParser(description='GAN-based unsupervised segmentation train')
        parser.add_argument('--args', type=str, default=None, help='json with all arguments')
        parser.add_argument('--out', type=str)
        parser.add_argument('--gan_weights', type=str, default=BIGBIGAN_WEIGHTS)
        parser.add_argument('--bg_direction', type=str)
        parser.add_argument('--device', type=int, default=0)
        #用于生成训练的显卡编号，本地只有一块卡因而写0
        parser.add_argument('--gen_devices', type=int, nargs='+', default=[0])
        parser.add_argument('--seed', type=int, default=2)
        parser.add_argument('--model', type=str, default=None)
        parser.add_argument('--z', type=str, default=None)
        parser.add_argument('--z_noise', type=float, default=0.0)
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--steps_per_validation', type=int, default=None)
        parser.add_argument('--n_steps', type=int, default=None)
        parser.add_argument('--epoch', type=int, default=None)
        parser.add_argument('--steps_per_log', type=int, default=None)
        parser.add_argument('--model_weight', type=str, default=None)
        parser.add_argument('--steps_per_weight_save', type=int, default=None)
        parser.add_argument('--syn_norm', type=bool, default=None)
        parser.add_argument('--record_file', type=int, default=None)
        parser.add_argument('--val_property_dir', type=str, default=None)
        parser.add_argument('--n_step_lis', type=int, nargs='+', default=None)
        parser.add_argument('--decay_lis', type=int, nargs='+', default=None)
        parser.add_argument('--image_property_dir', type=str, default=None)
        parser.add_argument('--image_root_dir', type=str, default=None)
        parser.add_argument('--val_root_dir', type=str, default=None)
        parser.add_argument('--steps_per_rate_decay', type=int, default=None)
        parser.add_argument('--epochs_per_log', type=int, default=None)
        parser.add_argument('--epochs_per_weight_save', type=int, default=None)
        parser.add_argument('--epochs_per_validation', type=int, default=None)
        parser.add_argument('--epochs_per_rate_decay', type=int, default=None)
        parser.add_argument('--dataset', type=str, default=None)
        parser.add_argument('--rate', type=float, default=None)
        parser.add_argument('--num_gpu', type=int, default=None)
        parser.add_argument('--hdf5', type=bool, default=None)

        #SPA
        parser.add_argument('--scg_fosc_th', type=float, default=0.1)
        parser.add_argument('--scg_sosc_th', type=float, default=0.5)
        parser.add_argument('--scg_order', type=int, default=2)
        parser.add_argument('--scg_so_weight', type=float, default=2)
        parser.add_argument('--scg_fg_th', type=float, default=0.1)
        parser.add_argument('--scg_bg_th', type=float, default=0.05)
        parser.add_argument('--scg_blocks', type=str, default='45')
        parser.add_argument('--scg', type=bool, default=True)
        parser.add_argument("--ram", type=float, default=0.1)
        parser.add_argument("--ram_start", type=int, default=0)
        parser.add_argument("--ra_loss_weight", type=float, default=0.5)
        parser.add_argument("--ram_th_bg", type=float, default=0.1)
        parser.add_argument("--ram_bg_fg_gap", type=float, default=0.2)

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
    info_record.record(param.__dict__,'log_detail')
    torch.random.manual_seed(param.seed)
    torch.cuda.set_device(param.device)
    
    init_seq = '{0} experment No: {1}  time : {2}'.format(\
            info_record.mode,info_record.id,info_record.time_sequence)
    print(init_seq)


    TrainOri = OriDataset(param.image_root_dir,param.image_property_dir,
                            hdf5=param.hdf5,
                         tsfm=transforms.Compose([transforms.Resize((256,256)), 
                                    transforms.RandomCrop(224), 
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    Trans(['Normalize','standard']),
                                     ]))
    Loader = DataLoader(TrainOri, batch_size=param.batch_size, shuffle=True, pin_memory=True ,num_workers=6)
    
    if 'CUB'in param.image_root_dir:
        model = eval(param.model).model(num_classes=200,args=param)
    else:
        model = eval(param.model).model(num_classes=1000,args=param)
    model = model.train().cuda()
    model = torch.nn.DataParallel(model.train().cuda(), range(param.num_gpu))

    # ipdb.set_trace()
    if param.model_weight != None:
        ckpt = torch.load(param.model_weight, map_location='cpu')
        try:
            model.load_state_dict(ckpt)
        except:
            _model_load(model, ckpt)
    
    
    # info_record.save_result_weight(model.state_dict())
    synthetic_score = train_segmentation(
        model, Loader, param, info_record, param.gen_devices)
    print('Synthetic data score: {}'.format(synthetic_score))

def _model_load(model, pretrained_dict):
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

def train_segmentation(model,CombLoader,params,info_record,gen_devices):
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

    # optimizer = torch.optim.Adam(model.parameters(), lr=params.rate)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, params.steps_per_rate_decay, params.rate_decay)
    
    optimizer = my_optim.get_finetune_optimizer(params, model)    
    model.module.loss_init()
    index = 0
    current_epoch = 0
    total_epoch = params.epoch
    while current_epoch < total_epoch:
        #加载权重and预处理
        # t0 = time.time()
        for index, dat in enumerate(CombLoader):
            # ipdb.set_trace()
            image,clas= dat[1],dat[2]
            image,clas = image.cuda(),clas.cuda()
            # t1 = time.time()
            index += 1
            # model.zero_grad()
            logits = model(image,clas)           
            loss = model.module.get_loss(logits,clas,current_epoch)
            #统计准确率
            cls_pre = logits[0].detach()
            model.module.loss_count(loss,cls_pre,clas)

            #优化
            optimizer.zero_grad()
            loss[0].backward()
            optimizer.step()
            # lr_scheduler.step()

            if index% 5000 == 0 and 'ILSVRC' in params.image_root_dir and index>0:
                loss_ct = model.module.loss_output()
                output = '{}% | step {}: loss:{}'\
                    .format(int(100.0 * current_epoch / params.current_epoch), current_epoch, loss_ct)
                print(output)
                info_record.record(output,'log_detail')
                # model.loss_reset()

            # current_epoch=1
            # res = my_optim.reduce_lr(params, optimizer, current_epoch)
            # if res:
            #     for g in optimizer.param_groups:
            #         out_str = 'Epoch:%d, %f\n'%(current_epoch, g['lr'])        
            # break

        current_epoch+=1
        res = my_optim.reduce_lr(params, optimizer, current_epoch)
        if res:
            for g in optimizer.param_groups:
                out_str = 'Epoch:%d, %f\n'%(current_epoch, g['lr'])
        
        
        # if index% params.steps_per_log == 0 and index>0:
        if current_epoch% params.epochs_per_log == 0:
            loss_ct = model.module.loss_output()
            output = '{}% | step {}: loss:{}'\
                .format(int(100.0 * current_epoch / params.epoch), current_epoch, loss_ct)
            print(output)
            info_record.record(output,'log_detail')
            model.module.loss_reset()

        #每steps_per_weight_save保存权重一次
        # if index% params.steps_per_weight_save == 0 and index>0:
        if current_epoch% params.epochs_per_weight_save == 0:
            info_record.save_result_weight(model.state_dict(),'{}_'.format(current_epoch))          

        #每steps_per_validation 用验证集检测一次
        # if index%params.steps_per_validation == 0:# and index>0:
        if current_epoch% params.epochs_per_validation == 0:
            model.module.eval()
            tsfm=transforms.Compose([ transforms.Resize((224,224)),
                transforms.ToTensor(), Trans(['Normalize','standard']) ])
            tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(28,28)])])

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

    
    model.eval()
    info_record.save_result_weight(model.state_dict())
    info_record.record('train complete as plan!')
    return 0

def check():
    param = TrainParams()
    print('run train with parameter:')
    print(param.__dict__)
    info_record = Train_Record(file=param.record_file)
    info_record.record(param.__dict__,'log_detail')
    torch.random.manual_seed(param.seed)
    torch.cuda.set_device(param.device)
    
    G = make_big_gan(param.gan_weights).eval().cpu()
    bg_direction = torch.load(param.bg_direction)
    mask_postprocessing = [connected_components_filter]
    params = [G,bg_direction,mask_postprocessing,param]
    devision = 'split'
    CombLoader = CombData(TrainDataset,MaskGenerator,devision,params)
    image,clas,mask = CombLoader[0]
    # print(image[0,:].max())
    # print(image[1,:].max())
    # print(image[2,:].max())
    # print(image[3,:].max(),'\n')

    # print(image[0,:].min())
    # print(image[1,:].min())
    # print(image[2,:].min())
    # print(image[3,:].min(),'\n')

    # print(image[0,:].mean())
    # print(image[1,:].mean())
    # print(image[2,:].mean())
    # print(image[3,:].mean())
    vis_image = visualize_tensor(image)
    vis_mask = visualize_tensor(mask.unsqueeze(1))
    
    vis_image.save_img(adaptive=False)
    vis_mask.save_img(adaptive=False)


if __name__ == '__main__':
    main()
    # check()