import os
import sys
import json
import torch
from scipy.stats import truncnorm
from concurrent.futures import Future
from threading import Thread
from torchvision.transforms import ToPILImage
import time
import cv2
import numpy as np
import ipdb
from torchvision.utils import save_image
# from utils.prepared_util import list2json,load_path_file



DEFULT_PATH = '/home/qxy/Desktop/BigGan/BigGAN/results/defult'
#.../BigGAN 根目录
PATH_FILE1 = os.path.abspath(os.path.split(__file__)[0]+'/../../weight/paths.json')
def load_path_file():
    with open(PATH_FILE1,'r') as f:
        path_file = json.load(f)
    return path_file
PATH_FILE = load_path_file()
ROOT_PATH = PATH_FILE['main_root']



def list2json(lists,json_file_path):
    '''input lists, json_file_path
    to write lists into json_file_path
    '''
    with open(json_file_path,'w') as f:
        f.write('[\n')
        ct=0
        for element in lists:
            element = json.dumps(element)
            ct+=1
            if ct == len(lists):
                f.write(element+'\n')
            else:
                f.write(element+',\n')
        f.write(']')

def _save_image(image, paths):
    return (cv2.imwrite(paths,image))

def _get_sequential_num():
    lt = time.localtime(time.time())
    a = '%02d'%int(lt.tm_mon)#月份
    b = '%02d'%int(lt.tm_mday)#日期
    localtime = '21'+a+b+'_'+'%02d'%lt.tm_hour+'_'+'%02d'%lt.tm_min+'%02d'%lt.tm_sec
    return localtime

def _make_defult_dir():
    dir_name = os.path.join(DEFULT_PATH,_get_sequential_num())
    os.makedirs(dir_name,exist_ok=True)
    return dir_name

def image_normalization(img):
    min_val = np.min(img)
    max_val = np.max(img)
    img_norm = (img - min_val)/(max_val - min_val)
    return img_norm

def save_synthetic(img_msk,length,path):
    '''target：保存生成的图片以及其mask

    '''
    img = img_msk[0]
    mask = img_msk[1]
    if path == 'default':
        paths = _make_defult_dir()

    for i in range(length):
        img_tmp = img[i].permute(1,2,0).cpu().numpy()
        save_image = image_normalization(img_tmp)*255
        save_paths = os.path.join(paths,str(i)+'.png')
        _save_image(save_image,save_paths)

    for i in range(length):
        mask_tmp = mask[i].squeeze().cpu().numpy()
        save_mask = image_normalization(mask_tmp)*255
        save_paths = os.path.join(paths,str(i)+'_mask.png')
        _save_image(save_mask,save_paths)

def save_generate(mask,length,path):
    '''target：保存生成的图片以及其mask

    '''

    if path == 'default':
        paths = _make_defult_dir()

    for i in range(length):
        mask_tmp = mask[i].squeeze().cpu().numpy().astype('float')
        save_mask = image_normalization(mask_tmp)*255
        save_paths = os.path.join(paths,str(i)+'_generate_mask.png')
        _save_image(save_mask,save_paths)

def formate_output(param):
    '''对参数param进行格式化输出:
    若param是dict，
    每个key:value占一行
    '''
    if type(param).__name__ == 'dict':
        for key, value in param.items():
            print('{:20s}: {}'.format(key,value))
    else:
        raise Exception('type of parameter is unformatable')

class Pretrained_Read(object):
    '''key word in info_dict:
    model_weight, optimzer_weight, current_iter, 
    '''
    def __init__(self):
        self.optimizer_load = False


    def add_info_dict(self,info_path):
        self.info_dict = torch.load(info_path, map_location='cpu')
        
        if 'model_weight' in self.info_dict.keys():
            self.model_weight =  self.info_dict['model_weight']
            self.optimizer_weight = self.info_dict['optimizer_weight']
            self.current_iter = self.info_dict['current_iter']
            self.scheduler_weight = self.info_dict['scheduler']
        else:
            self.model_weight =  self.info_dict
            self.optimizer_weight = None
            self.current_iter = None
            self.scheduler_weight = None

        

class Info_Record(object):
    '''用于记录test/train过程中产生的信息
    callable functions including: 
    record and format_record
    inputs:
    mode is options for [log, log_detail, both]
    '''
    def __init__(self,mode,file=1):
        '''生成必要的文件路径，
        生成实验序号，时间序号
        在log文件中添加实验头
        '''
        if mode == 'train':
            self.root_dir = os.path.join(ROOT_PATH,'results/train')
            self.weight_dir = os.path.join(ROOT_PATH,'weight/result_weight')
        else :
            self.root_dir = os.path.join(ROOT_PATH,'results/test')
        self.mode = mode
        if file == 1:
            self.log_dir = os.path.join(ROOT_PATH,'results/log.txt')
            self.log_detail_dir = os.path.join(ROOT_PATH,'results/log_detail.txt')
        elif file == 2:
            self.log_dir = os.path.join(ROOT_PATH,'results/log2.txt')
            self.log_detail_dir = os.path.join(ROOT_PATH,'results/log_detail2.txt')
        elif file == 3:
            self.log_dir = os.path.join(ROOT_PATH,'results/log3.txt')
            self.log_detail_dir = os.path.join(ROOT_PATH,'results/log_detail3.txt')
        
        self.id = self._gen_id() #int类型
        self.time_sequence = self._gen_time() #str类型
        self._initial()
        #self.file_dir = self._gen_record_file()
    
    
    #def record_param(param,record_file=None):
    def _initial(self):
        '''把编号写在log里
        格式：
        XXX(上一段内容)(空一行)

        mode experment No: Number  time : timesequence
        '''
        init_seq = '\n\n{0} experment No: {1}  time : {2}'.format(\
            self.mode,self.id,self.time_sequence)
        self._write(init_seq,self.log_dir)
        self._write(init_seq,self.log_detail_dir)

    def record(self,sequence,mode='both'):
        '''
        mode:‘both‘:在log和log detail上均记录
        ’log‘:仅在log.txt上记录；‘log_detail’：在log_detail.txt上记录
        '''
        if type(sequence).__name__ != 'str':
            sequence = str(sequence)
        if mode == 'both' or mode == 'log':
            self._write(sequence,self.log_dir)
        if mode =='both' or mode == 'log_detail':
            self._write(sequence,self.log_detail_dir)
        else:
            raise Exception('undinfed mode: {}'.format(mode))

    def _write(self,sequence,path):
        with open(path,'a') as f:
            f.write('\n'+sequence)
            
    def formate_record(self,param,mode='both'):
        '''对参数param进行格式化记录:
        若param是dict，
        每个key:value占一行
        mode:‘both‘:在log和log detail上均记录
        ’log‘:仅在log.txt上记录；
        ‘log_detail’：在log_detail.txt上记录
        '''
        if mode == 'both' or mode == 'log':
            self._formate_write(param,self.log_dir)
        if mode =='both' or mode == 'log_detail':
            self._formate_write(param,self.log_detail_dir)
        else:
            raise Exception('undinfed mode: {}'.format(mode))

    def _formate_write(self,param,path):
        if type(param).__name__ == 'dict':
            with open(path,'a') as f:
                for key, value in param.items():
                    f.write('\n{:20s}: {}'.format(key,value))
        else:
            raise Exception('type of parameter is unformatable')

    def _gen_time(self):
        '''获取每次实验的时间(精确到秒)构成
        '''
        lt = time.localtime(time.time())
        a = '%02d'%int(lt.tm_mon)#月份
        b = '%02d'%int(lt.tm_mday)#日期
        localtime = '21'+a+b+'_'+'%02d'%lt.tm_hour+'_'+'%02d'%lt.tm_min+'%02d'%lt.tm_sec
        return localtime

    def _gen_id(self):
        '''获取实验编号
        '''
        id_paths = os.path.join(ROOT_PATH,'results/current_id.txt')
        with open(id_paths,'r+') as f:
            Num = int(f.readlines()[0])
            write_num = Num + 1
            Num = '{:0>6d}'.format(Num)
        with open(id_paths,'w') as f:
            f.write(str(write_num))
        return Num


class Train_Record(Info_Record):
    def __init__(self,file):
        Info_Record.__init__(self,mode='train',file=file)

    def save_checkpoint(self, model, opt, scheduler, step,ct):
        '''保存checkpoint
        '''
        state_dict = {
        'model_weight': model.state_dict(),
        'optimizer_weight': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'current_iter': step}     
        save_dir = os.path.join(self.weight_dir, '{}_check_{}.pth'.format( str(self.id),str(ct) ))
        while os.path.exists(save_dir):
            save_dirr = save_dir.split('.pth')[0]+'_1.pth'
            print('opps we already have {}, current file will be saved into{}'
            .format(save_dir,save_dirr))
            save_dir = save_dirr
        torch.save(state_dict, save_dir)

    def save_result_weight(self,weight,ct=None):
        '''保存训练权重
        ct: 文件名后缀，通常由训练代数+phase组成，用来区分用一个id下不同阶段所保存的文件名称
        '''  
        if ct == None:
            save_dir = os.path.join(self.weight_dir, '{}_result.pth'.format( str(self.id)))
        else:
            save_dir = os.path.join(self.weight_dir, '{}_{}_result.pth'.format( str(self.id),ct ) )
        #to avoid id conflict bug, 
        #make sure each save wont cover previous weight
        while os.path.exists(save_dir):
            save_dirr = save_dir.split('.pth')[0]+'_1.pth'
            print('opps we already have {}, current file will be saved into{}'
            .format(save_dir,save_dirr))
            save_dir = save_dirr
        torch.save(weight,save_dir)
        return 



class dataset_file_load(object):
    #def __init__(self):
        #self.file_path = file_path
        #self.root_path = root_path

    def read_path_list(self,file_path,root_path,mask=False):
        '''
        由于mask路径要把.jpg改成.png,因此要做额外的处理
        '''
        relative_path = self._read_half_path(file_path,mask)
        absolute_path = self._concat_path_list(relative_path,root_path)
        
        return absolute_path

    def read_key_list(self,file_path,key):
        '''读取file_path中文件属性中关于key属性的list
        '''
        with open(file_path,'r') as f:
            property_list = json.load(f)
        
        key_list = []
        for dic in property_list:
            key_list.append(dic[key])    
        return key_list            

    def _read_half_path(self,file_path,mask):
        if file_path == None:
            file_path = self.file_path

        with open(file_path,'r') as f:
            property_list = json.load(f)
        path_list = []
        for img_property in property_list:
            if mask == False:
                path_list.append(img_property['path'])
            else:
                path_list.append(img_property['path'].replace('jpg','png'))
        return path_list

    def _concat_path_list(self,path_list,root_path):
        absolute_path=[]
        for paths in path_list:
            paths = os.path.join(root_path,paths)
            absolute_path.append(paths)
        return absolute_path

def save_clas_list(cls_list,id,dataset_name):
    paths = os.path.join(ROOT_PATH,'weight_result/cls_result')
    file_name = '{}_{}_{}'.format(id,dataset_name,len(cls_list[0]))
    paths = os.path.join(paths,file_name)
    os.makedirs(paths)
    top1_list,top5_list = cls_list
    top1_paths = os.path.join(paths,'classification1.json')
    top5_paths = os.path.join(paths,'classification5.json')
    list2json(top1_list,top1_paths)
    list2json(top5_list,top5_paths)
    return 0

