import os
import sys
import json
import torch
from scipy.stats import truncnorm
from concurrent.futures import Future
from threading import Thread
from torchvision.transforms import ToPILImage
import ipdb
from utils.prepared_util import list2json
import numpy as np
from metrics import ACC_record, cor2IoU,cor2IoU2,mask2cor,maskfilter,msk2IoU_pre,mask2cor_rgb
from utils.io_util import load_path_file

PATH_FILE = load_path_file()
ROOT_PATH = PATH_FILE['root_path']
def to_image(tensor, adaptive=False):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    if adaptive:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))
    else:
        tensor = (tensor + 1) / 2 #由-1～1映射到0.1
        tensor.clamp(0, 1)
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))

def to_gray_tensor(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor[0]
    tensor = 38*tensor[0,:]+75*tensor[1,:]+15*tensor[2,:]
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    return tensor

def make_noise(batch, dim, truncation=None):
    if isinstance(dim, int):
        dim = [dim]
    if truncation is None or truncation == 1.0:
        return torch.randn([batch] + dim)
    else:
        raise Exception('The code here have bugs so i annotated it')
        #return torch.from_numpy(truncnorm.rvs(-truncation, truncation, size=size)).to(torch.float)


def save_common_run_params(args):
    with open(os.path.join(args.out, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)
    with open(os.path.join(args.out, 'command.sh'), 'w') as command_file:
        command_file.write(' '.join(sys.argv))
        command_file.write('\n')


def run_in_background(func: callable, *args, **kwargs):
    """ run f(*args, **kwargs) in background and return Future for its outputs """
    future = Future()

    def _run():
        try:
            future.set_result(func(*args, **kwargs))
        except Exception as e:
            future.set_exception(e)

    Thread(target=_run).start()
    return future

class area_counter(object):
    '''bbox0:predict
    bbox1:real
    '''
    def __init__(self):
        self.area_list = np.array([0,0,0,0])
        self.length = np.array(0)
    
    def count(self,bbox0,bbox1):
        x0,y0,x1,y1 = bbox0
        x0_t,y0_t,x1_t,y1_t = bbox1
        ratio = ((x1-x0)*(y1-y0))/((x1_t-x0_t)*(y1_t-y0_t))
        self.length+=1
        if ratio<0.5:
            self.area_list[0]+=1
        elif 0.5<ratio and ratio<1:
            self.area_list[1]+=1
        elif 1<ratio and ratio<1.5:
            self.area_list[2]+=1
        else :
            self.area_list[3]+=1

    def __call__(self):
        return self.area_list/self.length




class acc_counter(object):
    def __init__(self):
        self.length = 0
        self.total = 0

    def reset(self):
        self.length = 0
        self.total = 0

    def __call__(self):
        return self.total/self.length
    
    def count(self,y_pred,y_tru):
        '''两个array变量
        '''
        self.total += (y_pred==y_tru).sum()
        self.length += len(y_pred)

class loss_counter(acc_counter):
    def __init__(self):
        super(loss_counter,self).__init__()

    def __call__(self):
        return super(loss_counter,self).__call__()
    
    def count(self,los):
        if torch.is_tensor(los):
            los=los.detach().item()
        self.total += los
        self.length += 1

class IoU_manager(object):
    '''inputs of __init__:
    thr_scale: list [], scheme of the filter threshold
    mark: indicate the way we generate bounding box, have three options:
    bbox_Psy, bbox, mask
    outputs:
    results of IoU of different threshold
    keys in acc_record
    '0.3'~'0.9';'miou','avg','box_v2'
    '''
    def __init__(self,thr_scale,mark='bbox',save_iou=False):
        self.thr_list = np.arange(thr_scale[0],thr_scale[1],thr_scale[2])
        acc_list = np.array([0 for x in self.thr_list])
        self.acc_record = ACC_record(acc_list,self.thr_list)
        self.mark = mark
        self.save_iou = save_iou
        if self.save_iou == True:
            self.iou_list=[]
    
    def update(self,mask_pre,gt):
        '''inputs:
        mask_pre: both (w,h) and (1,w,h) is acceptable
        gt: (w,h)
        outputs:
        cor_pre_list/mask_pre : predict location
        IoU_list : current img iou under different threshold 
        '''
        mark = self.mark
        mask_pre = mask_pre.squeeze()
        if mark == 'bbox_psy':
            cor_pre_list = mask2cor_rgb(mask_pre,self.thr_list)
            IoU_list = cor2IoU2(cor_pre_list,gt)
            self.acc_record.count(IoU_list)
            pre_list=cor_pre_list

        elif mark == 'bbox':
            cor_pre_list = mask2cor(mask_pre,self.thr_list)
            IoU_list = cor2IoU(cor_pre_list,gt)
            self.acc_record.count(IoU_list)
            pre_list=cor_pre_list

        elif mark == 'avg_bbox':
            thr_list_tmp = float(mask_pre.mean())+self.thr_list
            cor_pre_list = mask2cor(mask_pre,thr_list_tmp)
            IoU_list = cor2IoU(cor_pre_list,gt)
            self.acc_record.count(IoU_list)
            pre_list=cor_pre_list

        elif mark =='mask':
            mask_pre = maskfilter(mask_pre.squeeze(),self.thr_list)
            IoU_list = msk2IoU_pre(mask_pre,gt)
            self.acc_record.count(IoU_list)
            pre_list = mask_pre
        else:
            raise Exception('undefined mark: {}'.format(mark))

        if self.save_iou == True:
            self.iou_list.append(IoU_list)
        return pre_list,IoU_list

        

    def acc_output(self,disp_form=True):
        '''elements in acc_output:
        {'GT-Loc':[self.acc_list_05], '0.4':[acc_04,acc_04_thr], '0.5':[acc_05,acc_05_thr], \
                '0.6':[acc_06,acc_06_thr], '0.7':[acc_07,acc_07_thr], '0.8':[acc_08,acc_08_thr], '0.9':[acc_09,acc_09_thr], \
                'avg':[avg],'miou':[miou,miou_thr],'box_v2':[box_v2]}
        '''
        return self.acc_record.acc_output(disp_form)

    def save_iou_list(self,id,dataset_name):
        iou_list = self.iou_list
        paths = os.path.join(ROOT_PATH,'results/IoU_data')
        file_name = '{}_{}_{}.json'.format(id,dataset_name,len(iou_list))

        save_path = os.path.join(paths,file_name)
        list2json(iou_list,save_path)
        return 0


class Top_1_5(object):
    '''
    calculate top1/top5 list according to 
    clas results and iou_list 
    '''
    def __init__(self):
        self.IoU_list=[]
    
    def add(self,IoU_lis):
        self.IoU_list.append(IoU_lis)

    def save(self,paths):
        if len(self.IoU_list) == 0:
            raise Exception('the list is empty')
        else:
            list2json(self.IoU_list,paths)
    
    def reset(self):
        self.IoU_list = []

    def load(self,paths):
        if len(self.IoU_list) > 0:
            raise Exception('the list is not empty')
        else:
            with open(paths,'r') as f:
                self.IoU_list = json.load(f) 

    def pre_top1_top5(self,cls_paths):
        top1_list,top5_list = self._load_cls(cls_paths)
        IoU_list = self._select_iou_list(0.5)
        top1_loc = self._cal_cls_loc(IoU_list,top1_list)
        top5_loc = self._cal_cls_loc(IoU_list,top5_list)
        return top1_loc,top5_loc
    
    def _cal_cls_loc(self,IoU_list,top_list):
        ct = 0
        length = len(IoU_list)
        for i in range(length):
            if IoU_list[i]>0.5 and top_list[i]>50:
                ct+=1
        return ct/length


    def _select_iou_list(self,thr):
        IoU_list = np.array(self.IoU_list)
        acc_list = np.sum(IoU_list>0.5,axis=0)
        idx = np.argmax(acc_list)
        selected_iou = IoU_list[:,idx]
        return selected_iou


    def _load_cls(self,paths):
        cls1_path=os.path.join(paths,'data/classfication1.txt')
        cls5_path=os.path.join(paths,'data/classfication5.txt')
        with open(cls1_path,'r') as f:
            top1_list = json.load(f)
        with open(cls5_path,'r') as f:
            top5_list = json.load(f)
        return top1_list,top5_list