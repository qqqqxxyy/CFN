from PIL.Image import ImageTransformHandler
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from utils.utils import to_image,to_gray_tensor
from torchvision.utils import make_grid
import os
import cv2
import numpy as np
from torchvision.transforms import ToPILImage
import ipdb
from utils.prepared_util import load_path_file
PATH_FILE = load_path_file()
PATH = PATH_FILE['main_root']+'/results/pictures'
Idx_path = PATH_FILE['main_root']+'/results/pictures/0000000_current_id.txt'

class visualizer(object):
    def __init__(self):
        '''
            RGB格式：(N,3,w,h)
            Mask/htmp格式：(N,1,w,h)
        '''
    def save_htmp(self,tensor,paths=None,image = None,bbox_pre=None,bbox_gt=None):
        '''
        接受 N，1，w，h格式
        接受图片作为热力图底色
        bbox_pre:预测bbox用红色表示
            accepted for both [x0,y0,x1,y1] or [[x0,y0,x1,y1]]
        bbox_gt:真实bbox用绿色表示
            accepted for both [x0,y0,x1,y1] or [[x0,y0,x1,y1]]
        默认用相对值来决定高低
        '''
        # ipdb.set_trace()
        tensor = self._2htmpshape(tensor)
        paths = self._gen_paths(paths)
        #将第0维度扁平化
        tensor_grid = make_grid(tensor,nrow=1,padding=5)[0,:]

        np_grid = self._norm_np(self._tensor2np(tensor_grid))
        # ipdb.set_trace()
        np_grid = (np_grid*255).astype(np.uint8)

        htmp = cv2.applyColorMap(np_grid, cv2.COLORMAP_JET)
        if image != None:
            image = image.cpu().detach().numpy().transpose((1,2,0))
            image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            htmp = cv2.normalize(htmp.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            htmp = (htmp*0.6+image*0.4)*255
        if bbox_pre != None:
            if len(bbox_pre) == 1:
                bbox_pre = bbox_pre[0]
            cv2.rectangle(htmp , tuple(bbox_pre[:2]),tuple(bbox_pre[2:]),(0,0,255),2)
        if bbox_gt != None:
            if len(bbox_pre) == 1:
                bbox_pre = bbox_pre[0]
            cv2.rectangle(htmp , tuple(bbox_gt[:2]),tuple(bbox_gt[2:]),(0,255,0),2) 
        cv2.imwrite(paths,htmp)

    def save_graymp(self,tensor,paths=None,image = None):
        '''
        接受 N，1，w，h格式
        接受图片作为热力图底色
        不接受bbox_pre和bbox_gt
        bbox_pre:预测bbox用红色表示
        bbox_gt:真实bbox用绿色表示
        默认用相对值来决定高低
        '''

        tensor = self._2htmpshape(tensor)

        paths = self._gen_paths(paths)
        #将第0维度扁平化
        tensor_grid = make_grid(tensor,nrow=1,padding=5)[0,:]
        
        
        np_grid = self._norm_np(self._tensor2np(tensor_grid))

        htmp = (np_grid*255).astype(np.uint8)

        # htmp = cv2.applyColorMap(np_grid, cv2.COLORMAP_JET)
        # htmp=cv2.cvtColor(np_grid,cv2.COLOR_BGR2GRAY)
        if image != None:
            image = image.cpu().detach().numpy().transpose((1,2,0)).astype(np.uint8)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            htmp = cv2.normalize(htmp.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            htmp = (0.4-htmp*0.4+image*0.6)*255
        cv2.imwrite(paths,htmp)

    def save_image(self,tensor,paths=None,bbox=None,mask=None):
        '''
        N,1,w,h和N,3,w,h均接受
        paths 只接受相对文件夹，在picture casual之中创建文件
        以[path]or[path,filename]的方式接受路径
        bbox = [x0,y0,x1,y1]
        mask = 0-1 label
        '''
        if len(tensor.shape) ==3:
            tensor = tensor.unsqueeze(0)

        paths = self._gen_paths(paths)

        #首先扁平化
        tensor_grid = make_grid(tensor,nrow=1,padding=5)
        np_grid = self._norm_np(self._tensor2np(tensor_grid))

        np_grid = (np_grid.transpose((1,2,0))[:,:,[2,1,0]]*255).astype(np.uint8).copy()
        #为image添加mask
        if mask !=None:
            if torch.is_tensor(mask):
                mask_region = mask.cpu().detach().numpy()
                # ipdb.set_trace()
                #mask区域变红
                np_grid[mask_region[0,:,:],2] = np.array(255)#.astype(np.uint8).copy
                np_grid[mask_region[0,:,:],0] = np.array(0)
                np_grid[mask_region[0,:,:],1] = np.array(0)
                # (mask[:,:,0]).astype(np.uint8).copy()

        if bbox != None:
            cv2.rectangle(np_grid , tuple(bbox[:2]),tuple(bbox[2:]),(0,255,0,),2)

        cv2.imwrite(paths,np_grid)

    def _gen_paths(self,paths):
        '''inputs:
        paths: None: directly save image to /.../pictures/
        str: [path, name] path need to be complete paths, 
            if such path not exists, it will be created. 
        '''
        if paths==None:
            id  = self._gen_id()
            paths = os.path.join(PATH,id+'.png')
        else:
            if len(paths) == 1:
                id  = self._gen_id()
                filename  = id+'.png'
            else:
                filename = paths[1]
            rel_path = paths[0]
            paths = os.path.join(PATH,rel_path)
            if os.path.isdir(paths) == False:
                os.mkdir(paths)
            paths = os.path.join(paths,filename)
        return paths

    def _gen_id(self):
        '''获取实验编号s
        '''
        with open(Idx_path,'r+') as f:
            Num = int(f.readlines()[0])
            write_num = Num + 1
            Num = '{:0>6d}'.format(Num)
        with open(Idx_path,'w') as f:
            f.write(str(write_num))
        return Num

    def _tensor2np(self,tensor):
        return tensor.detach().cpu().numpy().astype('float')

    def _norm_np(self,np):
        # return cv2.normalize(np, None, 0.0, 1.0, cv2.NORM_MINMAX)
        image = (np - np.min()) / (np.max() - np.min())
        return image

    def _2htmpshape(self,tensor):
        '''
        W,H->1,1,W,H
        N,W,H-> N,1,W,Hhao
        '''
        if len(tensor.shape)==2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif len(tensor.shape)==3:
            tensor = tensor.unsqueeze(1)
        elif len(tensor.shape) != 4 or tensor.shape[1]!=1:
            print(tensor.shape)
            raise Exception('wrong form')
        return tensor