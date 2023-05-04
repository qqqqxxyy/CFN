#-*- coding: UTF-8 -*-
from random import randint
from re import I, U
from torchvision import transforms
from UNet.unet_model import UNet
#SegmentationInference, Threshold,resize_min_edge
import os
import sys
import argparse
import json
import torch
import cv2
from torchvision.transforms import ToPILImage, ToTensor, Resize
import numpy as np
#from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import ipdb
import torch.nn.functional as F
from utils.io_util import Info_Record,formate_output,Pretrained_Read
from utils.postprocessing import *
from utils.utils import to_image,IoU_manager
from utils.visual_util import visualizer
from metrics import ACC_record, cor2IoU,cor2IoU2,mask2cor,maskfilter,msk2IoU_pre,mask2cor_rgb
from UNet.unet_cam import *
from UNet.u2net import *
from UNet.unet_ssol import *
from compare_model.w_Psy_vgg import *
from gan_mask_gen import MaskGenerator, MaskSynthesizing, it_mask_gen #通过gan网络生成mask
from data import ObjLocDataset,Trans,TransBbox,SegFileDataset
from metrics import *
#model_metrics, IoU, accuracy, F_max,Localization,mask2IoU,cal_IoU
from BigGAN.gan_load import make_big_gan
from tqdm import tqdm
import pandas as pd
from ptflops import get_model_complexity_info
from utils.prepared_util import load_path_file
PATH_FILE = load_path_file()
UNET_PATH = '../weight/pretrained_weight/pre_u2net.pth'
class TestParams(object):

    def __init__(self):
        #固定参数
        self.batch_size= 1
        # self.mask_root_dir = 'CUB/segmentation'
        # self.image_root_dir = 'CUB/data'
        # self.image_property_dir = \
        #     'CUB/data_annotation/CUB_WSOL/test_list.json'
            # '/home/qxy/Desktop/datasets/CUB/data_annotation/CUB_WSOL/test_list.json'        
        self.model = 'UNet_cam'
        self.dataset = 'CUB'
        self.n_steps=None
        self.phase = 1
        self.val_property_dir = \
            'CUB/data_annotation/CUB_WSOL/test_list.json'
        self.image_root_dir = None
        self.image_property_dir = None
        #可以从命令行读入的参数
        parser = argparse.ArgumentParser(description='GAN-based unsupervised segmentation test')
        parser.add_argument('--model_weight', type=str, default=UNET_PATH)
        parser.add_argument('--n_steps', type=int, default=None)
        parser.add_argument('--model', type=str, default=None)
        # parser.add_argument('--image_property_dir', type=str, default=None)
        # parser.add_argument('--image_root_dir', type=str, default=None)
        parser.add_argument('--val_property_dir', type=str, default=None)
        # parser.add_argument('--val_root_dir', type=str, default=None)
        parser.add_argument('--length', type=int, default=500)
        parser.add_argument('--phase', type=int, default=None)
        parser.add_argument('--set', type=str, default=None)
        parser.add_argument('--dataset', type=str, default=None)
        parser.add_argument('--prediction_store',type=bool,default = False,help="Path of the image to load.",)

        args = parser.parse_args()       

        for key, val in args.__dict__.items():
            if val is not None:
                ##__dict__中存放self.XXX这样的属性，作为键和值存储
                #下面代码功能等同于将dict中所有元素作self.key = val的相应操作
                self.__dict__[key] = val 
        self._complete_formed_paths('dataset_root','mask_root_dir')
        # self._complete_paths('dataset_root','image_root_dir')
        # self._complete_paths('dataset_root','image_property_dir')
        self._complete_formed_paths('dataset_root','val_root_dir')
        self._complete_paths('dataset_root','val_property_dir')

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

    param = TestParams()
    print('run test with parameter:')
    formate_output(param.__dict__)
    info_record = Info_Record('test')
    info_record.record(param.__dict__,'both')
    init_seq = '{0} experment No: {1}  time : {2}'.format(\
            info_record.mode,info_record.id,info_record.time_sequence)
    print(init_seq)

    #real functioning part
    # check_try(param,info_record)
    check_predict_save(param,info_record)
    # check_visulization(param,info_record)

    info_record.record('test complete as plan!')


# def check_train(model,dataloader,length=None):
#     ''' blocker for validation during training process
#     '''
def check_train(model,dataloader,length=None):

    thr_scale = [0.01,0.90,0.02]
    IoU_man = IoU_manager(thr_scale,mark='bbox_psy')

    for sample in tqdm(enumerate(dataloader)):
        idx, image,clas, bbox,_= sample[1]
        image = image.squeeze(0)
        _,attmap = model(image.unsqueeze(0).cuda(non_blocking=True))
        # attmap = attmap[-1]
        # attmap = norm_att_map(attmap)

        attmap = F.interpolate(attmap.unsqueeze(dim=1), (image.size(1), image.size(2)), mode='bilinear', align_corners=False)
        attmap = attmap.detach().cpu().detach().numpy()
        # pre = attmap.squeeze()
        if torch.is_tensor(bbox[0][0]):
            bbox_lis = []
            for i in bbox:
                bbox_lis.append([float(x) for x in i])
            bbox = bbox_lis
        
        IoU_man.update(attmap,bbox)
        if idx == length:
            break
    info_dict = IoU_man.acc_output(disp_form=True)
    return info_dict



def check_try(param,info_record):

    target_ds = ObjLocDataset(param.val_root_dir,param.val_property_dir,
                tsfm=transforms.Compose([ transforms.Resize((224,224)),
                # tsfm=transforms.Compose([ transforms.Resize(224),
                    transforms.ToTensor(), Trans(['Normalize','standard']) ]),
                tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(224,224)]) ]) )

    tftype = ['rotation','odd']
    tfnum = [4, 3, 3, 2, 3, 5]
    net_work = eval(param.model)(vggconfig='vggcam16_bn',tftypes=tftype,tfnums=tfnum).cuda().eval()
    


    if 'Cub_VGG' in param.model_weight:
        checkpoint = torch.load(param.model_weight, map_location='cpu')
        net_work.load_state_dict(checkpoint['C_state_dict'])
    else:
        pretrain_info = Pretrained_Read()
        pretrain_info.add_info_dict(param.model_weight)
        net_work.load_state_dict(pretrain_info.model_weight)
    net_work.cuda().eval()
    

    thr_scale = [0.05,0.20,0.01]
    IoU_man = IoU_manager(thr_scale)  
    total =0.
    acc = 0.
    if param.n_steps == None:
        param.n_steps = len(target_ds)
    for t in tqdm(range(param.n_steps)):
        _, image,clas, bbox= target_ds[t]

        _,attmap = net_work(image.unsqueeze(0).cuda(non_blocking=True))



        attmap = F.interpolate(attmap.unsqueeze(dim=1), (image.size(1), image.size(2)), mode='bilinear', align_corners=False)
        attmap = attmap.cpu().detach().numpy()
        
        # pre = attmap.squeeze()
        # ipdb.set_trace()
        heatmap = intensity_to_rgb(np.squeeze(attmap), normalize=True).astype('uint8')
        gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
        th_val = 0.25 * np.max(gray_heatmap)
        _, th_gray_heatmap = cv2.threshold(gray_heatmap, int(th_val), 255, cv2.THRESH_BINARY)
        
        try:
            _, contours, _ = cv2.findContours(th_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except:
            contours, _ = cv2.findContours(th_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        
        if len(contours) != 0:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            estimated_box = [x, y, x + w, y + h]
            IOUr = calculate_IOU(bbox[0], estimated_box)
            total+=1
            if IOUr >=0.5:
                acc += 1
    print(acc/total)

def calculate_IOU(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def check_predict_save(param,info_record):
    target_ds = ObjLocDataset(param.val_root_dir,param.val_property_dir,
                tsfm=transforms.Compose([ transforms.Resize((224,224)),
                # tsfm=transforms.Compose([ transforms.Resize(224),
                    transforms.ToTensor(), Trans(['Normalize','standard']) ]),
                tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(224,224)]) ]) )

    tftype = ['rotation','odd']
    tfnum = [4, 3, 3, 2, 3, 5]
    net_work = eval(param.model)(vggconfig='vggcam16_bn',tftypes=tftype,tfnums=tfnum).cuda().eval()
    


    if 'Cub_VGG' in param.model_weight:
        checkpoint = torch.load(param.model_weight, map_location='cpu')
        net_work.load_state_dict(checkpoint['C_state_dict'])
    else:
        pretrain_info = Pretrained_Read()
        pretrain_info.add_info_dict(param.model_weight)
        try:
            net_work.load_state_dict(pretrain_info.model_weight)
        except :
            _model_load(net_work, pretrain_info.model_weight)

    net_work.cuda().eval()
    Id = info_record.id
    save_paths = '/home/qxy/Desktop/beta/results/Figure/{}_367_violin.xlsx'.format(Id)
    IoU_list = []

    thr_scale = [0.20,0.21,0.02]
    IoU_man = IoU_manager(thr_scale,mark='bbox_psy')
    if param.n_steps == None:
        param.n_steps = len(target_ds)
    SavePath = "/home/qxy/Desktop/g2amma/results/psulabel/Psy_VOC12_val.json"
    SaveLis = []
    for t in tqdm(range(param.n_steps)):
        _, image,clas, bbox,sizes= target_ds[t]
        _,attmap = net_work(image.unsqueeze(0).cuda(non_blocking=True))
        # attmap = attmap[-1]
        # attmap = norm_att_map(attmap)

        attmap = F.interpolate(attmap.unsqueeze(dim=1), (image.size(1), image.size(2)), mode='bilinear', align_corners=False)
        attmap = attmap.cpu().detach().numpy()
        # pre = attmap.squeeze()


        _, iou = IoU_man.update(attmap,bbox)


        if iou[0]>0.5:
            IoU_list.append(iou[0])

        # if param.prediction_store == True:
        pred = mask2cor_rgb(attmap,[0.20])
        x0,y0,x1,y1 = pred[0]
        w,h = sizes[0]
        x0,x1 = int(x0/224*w),int(x1/224*w)
        y0,y1 = int(y0/224*h),int(y1/224*h)
        pred = [[x0,y0,x1,y1]]
        for i in range(len(bbox)):
            x0,y0,x1,y1 = bbox[i]
            x0,x1 = int(x0/224*w),int(x1/224*w)
            y0,y1 = int(y0/224*h),int(y1/224*h)
            bbox[i] = [x0,y0,x1,y1]
        
        added = {"bbox":bbox,"predict":pred}
        SaveLis.append(added)


    info_dict = IoU_man.acc_output(disp_form=True)
    print(np.median(np.array(IoU_list)))

    # if param.prediction_store==True:
    print("store!")
    store_path = SavePath
    # os.path.join(prefix,param.set+'.json')
    with open(store_path,'w') as f:
        json.dump(SaveLis,f)

    # df = pd.DataFrame({'iou':IoU_list})
    # df.to_excel(save_paths,index=False,sheet_name='sheet1')
    info_record.record(thr_scale,'log_detail')
    info_record.formate_record(info_dict,'log_detail')
    print(info_dict)



def check(param,info_record):
    target_ds = ObjLocDataset(param.val_root_dir,param.val_property_dir,
                tsfm=transforms.Compose([ transforms.Resize((224,224)),
                # tsfm=transforms.Compose([ transforms.Resize(224),
                    transforms.ToTensor(), Trans(['Normalize','standard']) ]),
                tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(224,224)]) ]) )

    tftype = ['rotation','odd']
    tfnum = [4, 3, 3, 2, 3, 5]
    net_work = eval(param.model)(vggconfig='vggcam16_bn',tftypes=tftype,tfnums=tfnum).cuda().eval()
    


    if 'Cub_VGG' in param.model_weight:
        checkpoint = torch.load(param.model_weight, map_location='cpu')
        net_work.load_state_dict(checkpoint['C_state_dict'])
    else:
        pretrain_info = Pretrained_Read()
        pretrain_info.add_info_dict(param.model_weight)
        try:
            net_work.load_state_dict(pretrain_info.model_weight)
        except :
            _model_load(net_work, pretrain_info.model_weight)

    net_work.cuda().eval()
    Id = info_record.id
    save_paths = '/home/qxy/Desktop/beta/results/Figure/{}_367_violin.xlsx'.format(Id)
    IoU_list = []

    thr_scale = [0.20,0.21,0.02]
    IoU_man = IoU_manager(thr_scale,mark='bbox_psy')
    if param.n_steps == None:
        param.n_steps = len(target_ds)
    
    pseudo_list = []
    for t in tqdm(range(param.n_steps)):
        _, image,clas, bbox,sizes= target_ds[t]
        _,attmap = net_work(image.unsqueeze(0).cuda(non_blocking=True))
        # attmap = attmap[-1]
        # attmap = norm_att_map(attmap)

        attmap = F.interpolate(attmap.unsqueeze(dim=1), (image.size(1), image.size(2)), mode='bilinear', align_corners=False)
        attmap = attmap.cpu().detach().numpy()
        # pre = attmap.squeeze()


        _, iou = IoU_man.update(attmap,bbox)


        if iou[0]>0.5:
            IoU_list.append(iou[0])

        if param.prediction_store == True:
            pred = mask2cor_rgb(attmap,[0.20])
            x0,y0,x1,y1 = pred[0]
            w,h = sizes[0]
            x0,x1 = int(x0/224*w),int(x1/224*w)
            y0,y1 = int(y0/224*h),int(y1/224*h)
            pred = [x0,y0,x1,y1]
            pseudo_list.append([int(x) for x in pred])


    info_dict = IoU_man.acc_output(disp_form=True)
    print(np.median(np.array(IoU_list)))

    if param.prediction_store==True:
        print("store!")
        prefix = '/home/qxy/Desktop/datasets/CUB/data_annotation/'
        store_path = os.path.join(prefix,param.set+'.json')
        with open(store_path,'w') as f:
            json.dump(pseudo_list,f)

    # df = pd.DataFrame({'iou':IoU_list})
    # df.to_excel(save_paths,index=False,sheet_name='sheet1')
    info_record.record(thr_scale,'log_detail')
    info_record.formate_record(info_dict,'log_detail')
    print(info_dict)


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

def check_unet(param,info_record):
    target_ds = ObjLocDataset(param.image_root_dir,param.image_property_dir,
                tsfm=transforms.Compose([ transforms.Resize(128),
                    transforms.ToTensor(), Trans(['Normalize','standard']) ]),
                tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(128,)]) ]) )

    net_work = eval(param.model)().cuda().eval()
    net_work.load_state_dict(torch.load(param.model_weight, map_location='cpu'))
    net_work.cuda().eval()

    thr_scale = [0.25,0.50,0.01]
    IoU_man = IoU_manager(thr_scale)    
    for t in tqdm(range(param.n_steps)):
        _, image,clas, bbox= target_ds[t]
        mask_pre,cls = net_work(image.unsqueeze(0).cuda())
        # mask_pre = (mask_pre-mask_pre.min())/(mask_pre.max()-mask_pre.min())
        IoU_man.update(mask_pre,bbox)

    info_dict = IoU_man.acc_output(disp_form=True)
    info_record.record(thr_scale,'log_detail')
    info_record.formate_record(info_dict,'log_detail')
    print(info_dict)

def check_unet_mask(param,info_record):
    tftype = ['rotation','odd']
    tfnum = [4, 3, 3, 2, 3, 5]
    net_work = eval(param.model).model(vggconfig='vggcam16_bn',tftypes=tftype,tfnums=tfnum).cuda().eval()
    
    checkpoint = torch.load(param.model_weight, map_location='cpu')

    net_work.load_state_dict(checkpoint['C_state_dict'])

    net_work.cuda().eval()

    target_ds = SegFileDataset(param,crop=False,size=224)
    thr_scale = [0.19,0.20,0.01]
    IoU_man = IoU_manager(thr_scale)    
    for t in tqdm(range(param.n_steps)):
        image,mask_orao= target_ds[t]
        _,attmap = net_work(image.unsqueeze(0).cuda(non_blocking=True))
        attmap = attmap[-1]
        attmap = norm_att_map(attmap)

        attmap = F.interpolate(attmap.unsqueeze(dim=1), (image.size(1), image.size(2)), mode='bilinear', align_corners=False)
        attmap = attmap.cpu().detach().numpy()
        mask_pre = attmap.squeeze()
        bbox = mask2cor(mask_orao,0.5)
        # mask_pre = (mask_pre-mask_pre.min())/(mask_pre.max()-mask_pre.min())
        IoU_man.update(mask_pre,bbox)

    info_dict = IoU_man.acc_output(disp_form=True)
    info_record.record(thr_scale,'log_detail')
    info_record.formate_record(info_dict,'log_detail')
    print(info_dict)



def check_visulization(param,info_record):
    target_ds = ObjLocDataset(param.val_root_dir,param.val_property_dir,
                tsfm=transforms.Compose([ transforms.Resize((224,224)),
                # tsfm=transforms.Compose([ transforms.Resize(224),
                    transforms.ToTensor(), Trans(['Normalize','standard']) ]),
                tsfm_bbox=transforms.Compose([ TransBbox(['Resize',(224,224)]) ]) )

    tftype = ['rotation','odd']
    tfnum = [4, 3, 3, 2, 3, 5]
    net_work = eval(param.model)(vggconfig='vggcam16_bn',tftypes=tftype,tfnums=tfnum).cuda().eval()
    


    if 'Cub_VGG' in param.model_weight:
        checkpoint = torch.load(param.model_weight, map_location='cpu')
        net_work.load_state_dict(checkpoint['C_state_dict'])
    else:
        pretrain_info = Pretrained_Read()
        pretrain_info.add_info_dict(param.model_weight)
        try:
            net_work.load_state_dict(pretrain_info.model_weight)
        except :
            _model_load(net_work, pretrain_info.model_weight)

    net_work.cuda().eval()

    vis = visualizer()   
    thr_scale = [0.19]

    if param.n_steps == None:
        param.n_steps = len(target_ds)

    for t in tqdm(range(param.n_steps)):
        _, image,clas, bbox= target_ds[t]
        _,attmap = net_work(image.unsqueeze(0).cuda(non_blocking=True))
        # attmap = attmap[-1]
        # attmap = norm_att_map(attmap)

        attmap = F.interpolate(attmap.unsqueeze(dim=1), (image.size(1), image.size(2)), mode='bilinear', align_corners=False)
        # attmap = attmap.cpu().detach().numpy()
        # pre = attmap.squeeze()
        bbox_pre = mask2cor(attmap.squeeze(),thr_scale)
        bbox_real = [int(x) for x in bbox[0]]


        vis.save_htmp(attmap,image= image)
        # vis.save_htmp(attmap,image= image,bbox_pre = bbox_pre, bbox_gt= bbox_real)
        # vis.save_image(image,mask = attmap.squeeze(0)>0.19,bbox = bbox_pre[0])






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