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
from UNet.unet_model import UNet
from torchvision import transforms
from UNet.unet_cam import *
from UNet.u2net import *
from UNet.unet_ssol import *
from compare_model.w_Psy_vgg import *
from gan_mask_gen import MaskGenerator, MaskSynthesizing, it_mask_gen #通过gan网络生成mask
from data import OriDataset,CombData,ObjLocDataset,Trans,TransBbox
from metrics import ILSVRC_metrics, Localization, model_metrics, IoU, accuracy, F_max,ILSVRC_metrics
from BigGAN.gan_load import make_big_gan
from utils.postprocessing import connected_components_filter,\
    SegmentationInference, Threshold
from utils.utils import to_image, acc_counter
from utils.io_util import PATH_FILE, Train_Record,Pretrained_Read
from utils.prepared_util import list2json,load_path_file
from PsyNet_test import check_train
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
        self.mask_root_dir = 'CUB/segmentation'
        self.image_root_dir = 'CUB/data'
        self.image_property_dir = \
        'CUB/data_annotation/CUB_WSOL/train_list.json'
        self.syn_norm=False
        self.record_file=1
        self.val_property_dir = \
            'CUB/data_annotation/CUB_WSOL/val_list.json'
        self.val_root_dir = None
        self.o_value = 3.0
        self.dataset = 'CUB'

        #可以从命令行读入的参数
        parser = argparse.ArgumentParser(description='GAN-based unsupervised segmentation train')
        parser.add_argument('--args', type=str, default=None, help='json with all arguments')
        parser.add_argument('--out', type=str)
        parser.add_argument('--device', type=int, default=0)
        #用于生成训练的显卡编号，本地只有一块卡因而写0
        parser.add_argument('--gen_devices', type=int, nargs='+', default=[0])
        parser.add_argument('--seed', type=int, default=2)
        parser.add_argument('--model', type=str, default=None)
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
        parser.add_argument('--n_step_lis', type=int, default=None)
        parser.add_argument('--decay_lis', type=int, default=None)
        parser.add_argument('--batch_lis', type=int, default=None)
        parser.add_argument('--image_property_dir', type=str, default=None)
        parser.add_argument('--image_root_dir', type=str, default=None)
        parser.add_argument('--val_root_dir', type=str, default=None)
        parser.add_argument('--steps_per_rate_decay', type=int, default=None)
        parser.add_argument('--o_value', type=float, default=None)
        parser.add_argument('--rate', type=float, default=None)
        parser.add_argument('--dataset', type=str, default=None)

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
    torch.cuda.set_device(param.device)
    init_seq = '{0} experment No: {1}  time : {2}'.format(\
            info_record.mode,info_record.id,info_record.time_sequence)
    print(init_seq)


    TrainOri = OriDataset(param.image_root_dir,param.image_property_dir,
                         tsfm=transforms.Compose([transforms.Resize((256, 256)),
                                    transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
                                    transforms.ToTensor(),
                                    Trans(['Normalize','standard']),
                                     ]))
    TrainLoader =  DataLoader(TrainOri,batch_size=param.batch_lis,shuffle=True,\
        pin_memory=True,num_workers=6)

    tftype = ['rotation','odd']
    tfnum = [4, 3, 3, 2, 3, 5]
    tfval = {'T': 0.1, 'C': 0.3, 'S': 30, 'O': 3.0}

    tf_elements = [tftype,tfnum,tfval]
    model = eval(param.model)(vggconfig='vggcam16_bn',tftypes=tftype,tfnums=tfnum).train().cuda()
    
    if param.model_weight != None:
        if 'Cub_VGG' in param.model_weight:
            checkpoint = torch.load(param.model_weight, map_location='cpu')
            model.load_state_dict(checkpoint['C_state_dict'])
        else:
            pretrain_info = Pretrained_Read()
            pretrain_info.add_info_dict(param.model_weight)
            try:
                model.load_state_dict(pretrain_info.model_weight)
            except:
                _model_load(model, pretrain_info.model_weight)


    # if param.num_gpu != None:
    #     model = torch.nn.DataParallel(model.train().cuda(), range(param.num_gpu))


    param.n_steps = param.n_step_lis
    param.steps_per_rate_decay = param.decay_lis
    param.batch_size=param.batch_lis

    synthetic_score = train_segmentation(
        model, TrainLoader, param, info_record, param.gen_devices,tf_elements)

    
    info_record.record('train complete as plan!')
    print('Synthetic data score: {}'.format(synthetic_score))

def _model_load(model, pretrained_dict):
    '''this function can help to load the consistent part and ignore the rest part
    if parameters in checkpoint not totally fit model (this happens when train CAM-based 
    model with imagenet pretrained backbone)
    '''
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    print ("Weights cannot be loaded:")
    print ([k for k in model_dict.keys() if k not in pretrained_dict.keys()])
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

def supervision_generator(tfnums,tftypes,tfval,size):
    '''input: tfnums, tftypes, tfval, size
    output: mats, labels
    '''
    pi = torch.tensor(np.pi)
    rot_label = torch.tensor(np.random.choice(tfnums[0], size=(size,))).cuda( non_blocking=True)
    trs_lable = torch.tensor(np.random.choice(tfnums[1], size=(size,))).cuda( non_blocking=True)
    sh_label = torch.tensor(np.random.choice(tfnums[2], size=(size,))).cuda( non_blocking=True)
    hf_label = torch.tensor(np.random.choice(tfnums[3], size=(size,))).cuda( non_blocking=True)
    sc_label = torch.tensor(np.random.choice(tfnums[4], size=(size,))).cuda( non_blocking=True)
    od_lable = torch.tensor(np.random.choice(tfnums[5], size=(size,))).cuda( non_blocking=True)

    rot = (rot_label * (360.0/tfnums[0])).float()
    trs = ((trs_lable - (tfnums[1]//2)).float() * tfval['T']).float()
    sh = ((sh_label - 1) * tfval['S']).float()
    hf = (2 * (hf_label - 0.5)).float()
    sc = 1.0 - ((sc_label - 1.0).float() * tfval['C'])
    od = ((od_lable - (tfnums[5] // 2)).float() * tfval['O']).float()

    cosR = torch.cos(rot * pi / 180.0)
    sinR = torch.sin(rot * pi / 180.0)
    tanS = torch.tan(sh * pi / 180.0)

    rotmat = torch.zeros(size, 3, 3).cuda( non_blocking=True)
    trsmat = torch.zeros(size, 3, 3).cuda( non_blocking=True)
    shmat = torch.zeros(size, 3, 3).cuda( non_blocking=True)
    hfmat = torch.zeros(size, 3, 3).cuda( non_blocking=True)
    scmat = torch.zeros(size, 3, 3).cuda( non_blocking=True)
    odmat = torch.zeros(size, 3, 3).cuda( non_blocking=True)

    rotmat[:, 0, 0] = cosR
    rotmat[:, 0, 1] = -sinR
    rotmat[:, 1, 0] = sinR
    rotmat[:, 1, 1] = cosR
    rotmat[:, 2, 2] = 1.0

    trsmat[:, 0, 0] = 1.0
    trsmat[:, 0, 2] = trs
    trsmat[:, 1, 1] = 1.0
    trsmat[:, 1, 2] = trs
    trsmat[:, 2, 2] = 1.0

    shmat[:, 0, 0] = 1.0
    shmat[:, 0, 1] = tanS
    shmat[:, 1, 1] = 1.0
    shmat[:, 2, 2] = 1.0

    hfmat[:, 0, 0] = hf
    hfmat[:, 1, 1] = 1.0
    hfmat[:, 2, 2] = 1.0

    scmat[:, 0, 0] = sc
    scmat[:, 1, 1] = sc
    scmat[:, 2, 2] = 1.0

    odmat[:, 0, 0] = 1.0
    odmat[:, 0, 2] = od
    odmat[:, 1, 1] = 1.0
    odmat[:, 1, 2] = od
    odmat[:, 2, 2] = 1.0

    mats = []
    labels = []

    if 'odd' in tftypes:
        mats.append(odmat)
        labels.append(od_lable)
    if 'rotation' in tftypes:
        mats.append(rotmat)
        labels.append(rot_label)
    if 'translation' in tftypes:
        mats.append(trsmat)
        labels.append(trs_lable)
    if 'shear' in tftypes:
        mats.append(shmat)
        labels.append(sh_label)
    if 'hflip' in tftypes:
        mats.append(hfmat)
        labels.append(hf_label)
    if 'scale' in tftypes:
        mats.append(scmat)
        labels.append(sc_label)

    return mats, labels


def train_segmentation(model,CombLoader,params,info_record,gen_devices,tf_elements):
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
    tftype,tfnum,tfval = tf_elements

    optimizer = torch.optim.SGD(model.parameters(), 0.001, momentum=0.9, nesterov=True)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, params.steps_per_rate_decay, params.rate_decay)

    model.loss_init()
    index = 0

    model.train()
    # model.eval()
    # tsfm=transforms.Compose([ transforms.Resize((224,224)),
    #     transforms.ToTensor(), Trans(['Normalize','standard']) ])
    # tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(224,224)])])
    # val_ds = ObjLocDataset(params.val_root_dir,params.val_property_dir,tsfm,tsfm_bbox)
    # val_dl = torch.utils.data.DataLoader(val_ds,1,shuffle=False)
    # info = check_train(model,val_dl,length=50)
    # info_record.record(info,'log_detail')
    # print('0.5:',info['0.5'],';0.9:',info['0.9'],';mIoU:',info['miou'],';box_v2:',info['box_v2'])
    # model.train()

    while index<params.n_steps:
        #加载权重and预处理
        for _, dat in enumerate(CombLoader):
            _,image,clas = dat
            image,clas = image.cuda(),clas.cuda()
            index += 1
            batch = image.size(0)
            # data preperation
            mats, labels = supervision_generator(tfnum,tftype,tfval,size=batch)
            theta = mats[0]

            for matidx in range(1, len(mats)):
                theta = torch.matmul(theta, mats[matidx])
            theta = theta[:, :2, :]

            affgrid = F.affine_grid(theta, image.size(),align_corners=True).cuda(non_blocking=True)
            x_aff = F.grid_sample(image, affgrid, padding_mode='reflection',align_corners=True)

            model.zero_grad()

            c_logit, _ = model(x_aff)
            c_loss, record_loss = model.get_loss(c_logit,labels)

            #统计准确率
            model.loss_count(record_loss,c_logit,labels)

            #优化
            c_loss.backward()
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
                info_record.save_result_weight(model.state_dict(),'{}'.format(index))            


            #每steps_per_validation 用验证集检测一次
            if index%params.steps_per_validation == 0:# and index>0:
                model.eval()
                tsfm=transforms.Compose([ transforms.Resize((224,224)),
                    transforms.ToTensor(), Trans(['Normalize','standard']) ])
                tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(224,224)])])

                val_ds = ObjLocDataset(params.val_root_dir,params.val_property_dir,tsfm,tsfm_bbox)
                val_dl = torch.utils.data.DataLoader(val_ds,1,shuffle=False)
                info = check_train(model,val_dl,length=500)
                info_record.record(info,'log_detail')
                print('0.5:',info['0.5'],';0.9:',info['0.9'],';mIoU:',info['miou'],';box_v2:',info['box_v2'])
                model.train()

            if index >= params.n_steps:
                break
    
    model.eval()
    info_record.save_result_weight(model.state_dict())

    return 0

if __name__ == '__main__':
    main()