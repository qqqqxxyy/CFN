#-*- coding: UTF-8 -*-
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
from data import SegFileDataset
from UNet.unet_model import UNet
from UNet.unet_cam import *
from gan_mask_gen import MaskGenerator, MaskSynthesizing, it_mask_gen #通过gan网络生成mask
from data import TrainDataset, FiledDataset,CombData,TestDataset
from metrics import ILSVRC_metrics, Localization, model_metrics, IoU, accuracy, F_max,ILSVRC_metrics
from BigGAN.gan_load import make_big_gan
from utils.postprocessing import connected_components_filter,\
    SegmentationInference, Threshold
from utils.utils import to_image, acc_counter
from utils.io_util import Train_Record
from utils.visual_util import visualize_tensor
import time
DEFAULT_EVAL_KEY = 'id'
THR_EVAL_KEY = 'thr'
SEGMENTATION_RES = 128
BIGBIGAN_WEIGHTS = '../weight_result/weights/BigBiGAN_x1.pth'
LATENT_DIRECTION = '../weight_result/weights/bg_direction.pth'


MASK_SYNTHEZ_DICT = {
    'lighting': MaskSynthesizing.LIGHTING,
    'mean_thr': MaskSynthesizing.MEAN_THR,
}
PATHS_FILE='../weight_result/paths.json'

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
        self.mask_root_dir = 'CUB/segmentation'
        self.image_root_dir = 'CUB/data'
        self.image_property_dir = \
        'CUB/data_annotation/CUB_WSOL/train_list.json'
        self.syn_norm=False
        self.record_file=1
        self.val_property_dir = \
            'CUB/data_annotation/CUB_WSOL/val_list.json'
        self.val_root_dir = None


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
        parser.add_argument('--steps_per_log', type=int, default=None)
        parser.add_argument('--model_weight', type=str, default=None)
        parser.add_argument('--steps_per_weight_save', type=int, default=None)
        parser.add_argument('--syn_norm', type=bool, default=None)
        parser.add_argument('--record_file', type=int, default=None)
        parser.add_argument('--val_property_dir', type=str, default=None)
        parser.add_argument('--n_step_lis', type=int, nargs='+', default=None)
        parser.add_argument('--decay_lis', type=int, nargs='+', default=None)
        parser.add_argument('--batch_lis', type=int, nargs='+', default=None)
        parser.add_argument('--image_property_dir', type=str, default=None)
        parser.add_argument('--image_root_dir', type=str, default=None)
        parser.add_argument('--val_root_dir', type=str, default=None)
        parser.add_argument('--steps_per_rate_decay', type=int, default=None)

        args = parser.parse_args()
        self._add_parameter(args.__dict__)
        self._add_parameter(kwargs)

        self._complete_paths('dataset_root','mask_root_dir')
        self._complete_paths('dataset_root','image_root_dir')
        self._complete_paths('dataset_root','image_property_dir')
        self._complete_paths('dataset_root','val_property_dir')
        self._complete_paths('dataset_root','val_root_dir')

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
        with open(PATHS_FILE) as f:
            path_dict = json.load(f)
            root_path = path_dict[rootpath]
        self.__dict__[abspath] = os.path.join(root_path,abss)        

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


    G = make_big_gan(param.gan_weights).eval().cpu()
    bg_direction = torch.load(param.bg_direction)
    mask_postprocessing = [connected_components_filter]
    params = [G,bg_direction,mask_postprocessing,param]
    devision = 'split'
    CombLoader = CombData(TrainDataset,MaskGenerator,devision,params)

    model = eval(param.model)().train().cuda()
    if param.model_weight != None:
        model.load_state_dict(torch.load(param.model_weight, map_location='cpu'))

    # BATCH_SIZE = param.batch_size
    # #phase=2
    # #提前申请足够的内存
    # param.n_steps = 10
    # param.steps_per_rate_decay = 10
    # model.phase = 2
    # param.batch_size=40
    # CombLoader = CombData(TrainDataset,MaskGenerator,devision,params)
    # synthetic_score = train_segmentation(
    #     model, CombLoader, param, info_record, param.out, param.gen_devices)

    #phase=1
    param.n_steps = param.n_step_lis[0]
    param.steps_per_rate_decay = param.decay_lis[0]
    model.phase = 1
    param.batch_size=param.batch_lis[0]
    CombLoader = CombData(TrainDataset,MaskGenerator,devision,params)
    synthetic_score = train_segmentation(
        model, CombLoader, param, info_record, param.out, param.gen_devices)

    #phase=2
    # param.rate=0.01
    param.n_steps = param.n_step_lis[1]
    param.steps_per_rate_decay = param.decay_lis[1]
    model.phase = 2
    ##为了节省gpu第二阶段batchsize降成50
    param.batch_size=param.batch_lis[1]
    CombLoader = CombData(TrainDataset,MaskGenerator,devision,params)
    synthetic_score = train_segmentation(
        model, CombLoader, param, info_record, param.out, param.gen_devices)

    # phase=3
    # 固定分类器的权重
    for p in model.classifier.parameters():
        p.requires_grad = False

        
    param.n_steps = param.n_step_lis[2]
    param.steps_per_rate_decay = param.decay_lis[2]
    model.phase = 3
    param.batch_size=param.batch_lis[2]
    CombLoader = CombData(TrainDataset,MaskGenerator,devision,params)
    synthetic_score = train_segmentation(
        model, CombLoader, param, info_record, param.out, param.gen_devices)
    print('Synthetic data score: {}'.format(synthetic_score))


def read_train_np(paths):
    '''加载z向量和每一个z向量所对应的类别
    '''
    zs_path = paths
    zs_cls_path = paths.split('.npy')[0]+'_cls.npy'
    zs = torch.from_numpy(np.load(zs_path))
    zs_cls = torch.from_numpy(np.load(zs_cls_path))
    return zs, zs_cls

def train_segmentation(model,CombLoader,params,info_record,out_dir,gen_devices):
    '''target:训练model
    input:
    G: BigGAN生成模型
    bg_direction: 纺射矩阵
    mdoel: 分割模型
    params: 训练参数
    out_dir: 保存文件地址
    gen_devices: gpu数量
    '''
    model.train()
    os.makedirs(out_dir,exist_ok=True)
    params.batch_size = params.batch_size // len(gen_devices)

    optimizer = torch.optim.Adam(model.parameters(), lr=params.rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, params.steps_per_rate_decay, params.rate_decay)
    

    model.loss_init()
    index = 0
    while True:
        #加载权重and预处理
        # t0 = time.time()
        image,clas,mask = CombLoader[index]
        # t1 = time.time()
        index += 1
        model.zero_grad()

        logits = model(image,clas)
        cls_pre = logits[0].detach()
        
        loss = model.get_loss(logits,clas,mask,index)
        #统计准确率
        model.loss_count(loss,cls_pre,clas)

        #优化
        loss[0].backward()
        optimizer.step()
        lr_scheduler.step()
        # t2 = time.time()
        # print(t1-t0,' ',t2-t1)
        #每 steps_per_log 进行记录一次
        if index% params.steps_per_log == 0 and index>0:
            
            loss_ct = model.loss_output()
            output = '{}% | step {}: loss:{}'\
                .format(int(100.0 * index / params.n_steps), index, loss_ct)
            print(output)
            info_record.record(output,'log_detail')
            model.loss_reset()

        #每steps_per_weight_save保存权重一次
        if index% params.steps_per_weight_save == 0 and index>0:
            info_record.save_result_weight(model.state_dict(),'{}_{}'.format(index,model.phase))            

        #每steps_per_validation 用验证集检测一次
        if index%params.steps_per_validation == 0:# and index>0:
            model.eval()
            if 'CUB' in params.image_root_dir:
                val_ds = SegFileDataset(params,crop=True,size=(128,128),val=True)
                val_dl = torch.utils.data.DataLoader(val_ds,1,shuffle=False)
                
                outdict = model_metrics(Threshold(model, thr=0.5, resize_to=128), val_dl, \
                                            stats=(IoU, accuracy,Localization), n_steps = 500)
            elif 'ILSVRC' in params.image_root_dir:
                val_ds = TestDataset(params.val_root_dir,params.val_property_dir)
                val_dl = torch.utils.data.DataLoader(val_ds,1,shuffle=False)
                outdict = ILSVRC_metrics(Threshold(model,thr=0.5,resize_to=128),val_dl,n_steps=1000)
                # outdict = model_metrics(Threshold(model,thr=0.5,resize_to=128),val_dl, \
                #                         stats=(bbox_loc),n_steps=1000)

            output = 'Thresholded:\n{}'.format( outdict )
            print(output)
            model.train()
        if index >= params.n_steps:
            break
    
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
    # ipdb.set_trace()
    vis_image = visualize_tensor(image)
    vis_mask = visualize_tensor(mask.unsqueeze(1))
    
    vis_image.save_img(adaptive=False)
    vis_mask.save_img(adaptive=False)


if __name__ == '__main__':
    main()
    # check()