#-*- coding: UTF-8 -*-
import os

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from utils.postprocessing import resize
from torchvision import transforms
from PIL import Image
from utils.io_util import dataset_file_load
import torch
import numpy as np
import copy
import ipdb 
import json
import h5py
IMAGENET_PATH = '__local_imagenet_train_dir__'


def Random_noise(mask,n_prob):
    '''mask: tensor, (n,128,128) 
    n_prob: float (0,1), the percentage of noise
    '''
    n = mask.shape[0]
    a = mask.shape[1]
    b = mask.shape[2]
    noise_tensor = torch.rand((n,a,b))
    mask[noise_tensor<n_prob]=1-2*mask[noise_tensor<n_prob]
    return mask

class Trans(object):
    '''
    crop_style:[label,number]
    [None,None]; [central_crop,None]; [center_crop,size];
    [random_crop,size]; tencrop;
    '''
    def __init__(self,Trans_style):
        self.trans_style = Trans_style[0]
        self.prop = Trans_style[1]
    
    def __call__(self,img):
        if self.trans_style == 'Central_crop':
            dims = img.size
            crop = transforms.CenterCrop(min(dims[0], dims[1]))
            img = crop(img)
        elif self.trans_style == 'Normalize':
            img = self._normalize(img)
        return img
    
    def _normalize(self,img):
        if self.prop == 'standard':
            mean_vals = [0.485, 0.456, 0.406]
            std_vals = [0.229, 0.224, 0.225]
        elif self.prop == '0.5':
            mean_vals = [0.5, 0.5, 0.5]
            std_vals = [0.5, 0.5, 0.5]
        else:
            raise Exception('undefined prop {}'.format(self.prop))
        img = transforms.Normalize(mean_vals,std_vals)(img)
        return img

class TransBbox(object):
    def __init__(self,Trans_style):
        self.trans_style = Trans_style[0]
        self.prop = Trans_style[1]
    
    def __call__(self,sample):
        bbox,size = sample
        if self.trans_style == None:
            pass
        elif self.trans_style == 'Central_crop':
            bbox,size = self._central_crop(bbox,size)
        elif self.trans_style == 'Resize':
            bbox,size = self._resize(bbox,size)
        else:
            raise Exception('undefined prop {}'.format(self.trans_style))
        return [bbox,size]

    def _central_crop(self,bbox_lis,size):
        out_bbox_lis=[]
        for bbox in bbox_lis:
            w,h = size
            x0,y0,x1,y1=bbox
            if w>=h :
                d = 1/2*(w-h)
                x0,x1 = x0-d,x1-d
                x0 = max(0,x0)
                x1 = min(h,x1)
                w=h
            else:
                d = 1/2*(h-w)
                y0,y1 = y0-d,y1-d
                y0 = max(0,y0)
                y1 = min(w,y1)
                h=w
            out_bbox_lis.append([x0,y0,x1,y1])
        return out_bbox_lis,[w,h]
    
    def _resize(self,bbox_lis,size):
        '''
        inputs: 
        size: (w,h) or (w)
        '''
        out_bbox_lis=[]
        for bbox in bbox_lis:
            w,h = size
            x0,y0,x1,y1 = bbox
            
            if len(self.prop) == 2:
                W,H = self.prop
                x0,x1 = W/w*x0,W/w*x1
                y0,y1 = H/h*y0,H/h*y1
            elif len(self.prop) ==1:
                w,h = size
                if w<h:
                    W = self.prop[0]
                    ratio = self.prop[0]/w
                    H = ratio*h
                else:
                    H = self.prop[0]
                    ratio = self.prop[0]/h
                    W = ratio*w
                x0,x1,y0,y1 = x0*ratio,x1*ratio,y0*ratio,y1*ratio
            else:
                raise Exception('unacceptable length of size : len={}'.format(len(self.prop)))
            out_bbox_lis.append([x0,y0,x1,y1])
        return out_bbox_lis,[W,H]

class TransMask(object):
    '''
    structure of sample
    '''
    def __init__(self,Tran_style):
        self.trans_style = Tran_style[0]
        self.prop = Tran_style[1]

    def __call__(self,sample):
        if self.trans_style == 'Resize':
            sample = self._resize(sample)
        elif self.trans_style == 'RandomCrop':
            seed = np.random.randint(2147483647)
            sample = self._random_crop(sample,seed)
        elif self.trans_style == 'ToTensor':
            sample = self._to_tensor(sample)
        elif self.trans_style == 'Normalize':
            sample = self._normalize(sample)
        else:
            raise Exception('Wrong transformation instruct {}'.format(self.trans_style))
        return sample
    
    def _resize(self,sample):
        image,mask = sample
        size = self.prop
        image_t = transforms.Resize(size)(image)
        mask_t = transforms.Resize(size)(mask)
        return [image_t, mask_t]

    def _random_crop(self,sample,seed):
        image,mask = sample
        size = self.prop
        torch.random.manual_seed(seed)
        image_t = transforms.RandomCrop(size)(image)
        torch.random.manual_seed(seed)
        mask_t = transforms.RandomCrop(size)(mask)
        return [image_t, mask_t]
    
    def _to_tensor(self,sample):
        image,mask = sample
        image_t = transforms.ToTensor()(image)
        mask_t = transforms.ToTensor()(mask)  
        return [image_t, mask_t]

    def _normalize(self,sample):
        image,mask = sample
        if self.prop == 'standard':
            mean_vals = [0.485, 0.456, 0.406]
            std_vals = [0.229, 0.224, 0.225]
        elif self.prop == '0.5':
            mean_vals = [0.5, 0.5, 0.5]
            std_vals = [0.5, 0.5, 0.5]
        else:
            raise Exception('undefined prop {}'.format(self.prop))
        image_t = transforms.Normalize(mean_vals,std_vals)(image)
        mask_t = mask
        return [image_t, mask_t]

class TrainDataset(Dataset):
    '''cls模式中的训练集
    '''
    def __init__(self,images_root,image_prop):
        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
        tsfm_train = transforms.Compose([transforms.Resize((156,156)), 
                                     transforms.RandomCrop(128), #128
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                    transforms.Normalize((.5,.5,.5),(.5,.5,.5)),
                                     ])
        self.dataset = FiledDataset(images_root,image_prop,transform=tsfm_train)
        load_tool = dataset_file_load()
        self.clas_list = torch.tensor(load_tool.read_key_list(image_prop,'class'))-1

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        '''img tensor (3,128,128)
        clas tensor (1,)
        '''
        img = self.dataset[idx]
        clas  = self.clas_list[idx]
        return img,clas

class OriDataset(Dataset):
    '''the base function of dataset, including: 
    image loading, class loading, tsfm of image
    input:
    images_root: string variable, the path of dataset
    image_prop: string variable, the path of property: json file
    tsfm: dict funtion, {'resize':tuple, 'normalize': /; 'crop':/}
    output:
    idx: the index is called for
    img: tensor (w,h,3)
    clas: tensor (0)
    '''
    def __init__(self,images_root,image_prop,tsfm=None,mask=False,hdf5=False):
        self.load_tool = dataset_file_load()
        #read image path list
        
        self.hdf5 = hdf5
        if self.hdf5 == True:
            #images_root: 索取要读取的目录文件
            #images_path_list: 文件名
            self.image_path_list = self.load_tool._read_half_path(image_prop,mask)
            self.images_root = images_root+'.hdf'
            # self.images_root = '/mnt/usb/Dataset relate/ILSVRC_hdf5/data.hdf'
        else:
            self.image_path_list = self.load_tool.read_path_list(image_prop,images_root,mask)
        #read class
        #-1 because in property.json, the class is started with 1
        #针对VOC2012没有保存类别标签,进行伪造类别标签(不影响localization结果)
        try :
            self.clas_list = torch.tensor(self.load_tool.read_key_list(image_prop,'class'))-1
        except:
            self.clas_list = torch.tensor( [0] * len(self.image_path_list) )
            
        self.transform  = tsfm


    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self,idx):
        if self.hdf5 == True:
            img = self._read_hdf5(idx)
        else:
            img = Image.open(self.image_path_list[idx])

        if self.transform != None:
            img = self.transform(img.convert('RGB'))
        clas  = self.clas_list[idx]

        return idx, img, clas

    def _read_hdf5(self,idx):
        image_name = self.image_path_list[idx]
        with h5py.File(self.images_root,'r') as hdf:
            img =  np.array(hdf.get(image_name))
            img = Image.fromarray(img)
        return img

class ObjLocDataset(OriDataset):
    'extra output bounding box based on OriDataset'    
    def __init__(self,images_root,image_prop,tsfm,tsfm_bbox,hdf5=False):
        super(ObjLocDataset, self).__init__(images_root,image_prop,tsfm,hdf5=hdf5)
        self.bbox_list = self.load_tool.read_key_list(image_prop,'bbox')
        self.size_list = self.load_tool.read_key_list(image_prop,'size')
        self.tsfm_bbox = tsfm_bbox

    def __getitem__(self,idx):
        '''
        output:
        image: tensor (3,w,h)
        clas: tensor (0)
        bbox: float list [x0,y0,x1,y1]
        '''
        idx,img,clas = super(ObjLocDataset, self).__getitem__(idx)
        bbox,_ = self.tsfm_bbox([self.bbox_list[idx],self.size_list[idx][0]])
        return idx,img,clas,bbox,self.size_list[idx]

class ObjMaskDataset(OriDataset):
    'extra output mask based on OriDataset'
    def __init__(self,images_root,mask_root,image_prop,tsfm):
        self.Image = OriDataset(images_root,image_prop)
        self.Mask = OriDataset(mask_root,image_prop,mask=True)
        self.tsfm = tsfm

    def __len__(self):
        return len(self.Image)

    def __getitem__(self, idx):
        _,image,_ = self.Image.__getitem__(idx)
        image = image.convert('RGB')
        _,mask,_ = self.Mask.__getitem__(idx)
        element = [image, mask]
        image, mask = self.tsfm(element)
        return idx, image, mask[0]



def central_crop(x):
    dims = x.size
    crop = transforms.CenterCrop(min(dims[0], dims[1]))
    return crop(x)


def _id(x):
    return x


def _filename(path):
    return os.path.basename(path).split('.')[0]


def _numerical_order(files):
    return sorted(files, key=lambda x: int(x.split('.')[0]))


class FiledDataset(Dataset):
    def __init__(self,images_root,image_prop,mask=False,transform=None):
        '''
        测试数据集
        input:
        images_root : 文件信息的根路径，和相对路径组合形成绝对路径 image_path_list
        image_prop : 文件信息的相对路径
        '''
        load_tool = dataset_file_load()
        self.image_path_list = load_tool.read_path_list(image_prop,images_root,mask)
        #mask_path_list = load_tool.read_path_list(image_prop,masks_root)
        self.transform = transform
    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, item):
        img = Image.open(self.image_path_list[item])
        img = img.convert('RGB')

        #shape = img.shape

        if self.transform is not None:
            return self.transform(img)
        else:
            return img

class TransformedDataset(Dataset):
    def __init__(self, source, transform, img_index=0):
        self.source = source
        self.transform = transform
        self.img_index = img_index

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        out = self.source[index]
        if isinstance(out,tuple):
            return self.transform(out[self.img_index]), out[1 - self.img_index]
        else:
            return self.transform(out)

class SegFileDataset(Dataset):
    def __init__(self,param,crop=True,size=None,mask_thr=0.5,val=False):
        self.mask_thr = mask_thr
        mask_root = param.mask_root_dir
        if val == False:
            img_root = param.image_root_dir
            img_property = param.image_property_dir
        else:
            img_root = param.val_root_dir
            img_property = param.val_property_dir


        images_ds = FiledDataset(img_root, img_property)
        masks_ds = FiledDataset(mask_root, img_property,mask=True)

        mean_vals = [0.485, 0.456, 0.406]
        std_vals = [0.229, 0.224, 0.225]
        shift_to_zero = lambda x: 2 * x - 1
        resize = transforms.Compose([
            central_crop if crop else _id,
            transforms.Resize(size) if size is not None else _id,
            transforms.ToTensor(),
            #shift_to_zero,
            # transforms.Normalize((.5,.5,.5),(.5,.5,.5))
            transforms.Normalize(mean_vals,std_vals)
            ])
        resize2 = transforms.Compose([
            central_crop if crop else _id,
            transforms.Resize(size) if size is not None else _id,
            transforms.ToTensor()
            ])
        self.images_ds = TransformedDataset(images_ds, resize)
        self.masks_ds = TransformedDataset(masks_ds, resize2)

    def __len__(self):
        return len(self.images_ds)

    def __getitem__(self, index):
        mask = self.masks_ds[index] >= self.mask_thr  #对于真实图像的label，也用这种粗糙的0.5iou方式，不过也无可厚非了
        return (self.images_ds[index], mask[0])


class CombData_pre(object):
    def __init__(self,dataL,maskG,devision,param):
        '''param结构：
        list： param[0]=G; param[1]=bg_direction, param[2]=mask_postprocessing
        param[3] = param
        '''
        
        self.dataL = self._make_dataL(dataL,param,devision)
        # self.loader = iter(self.dataL)
        self.maskG = self._make_maskG(maskG,param,devision)
        self.devision = devision

    def __getitem__(self,idx):
        '''对split情形的结合选择：
        [真实,合成]
        '''
        if self.devision == 'real':
            _,image_r, clas = self.for_dataL(idx)
            return image_r.cuda(), clas.cuda(), torch.tensor([])        
        elif self.devision == 'synthetic':
            image_s, mask = self.for_maskG(idx)
            return image_s, torch.tensor([]) ,mask
        elif self.devision == 'split':
            _,image_r, clas = self.for_dataL(idx)
            image_s, mask = self.for_maskG(idx)
            image = torch.cat((image_r.cuda(),image_s),0)
            return image, clas.cuda(), mask
        else:
            raise Exception('wrong devision coding {}'.format(self.devision))
    
    def for_dataL(self,idx):
        '''读取真实数据image_r, clas
        '''
        if self.dataL == None:
            return None,None
        else:
            if idx%len(self.dataL) == 0:
                self.loader = iter(self.dataL)
            return next(self.loader)

    def _make_dataL(self,dataL,param,devision):
        P = param[3]
        if devision == 'synthetic':
            return None
        elif devision == 'split':
            batch_size = int(P.batch_size/2)
        else:
            batch_size = P.batch_size
        # trainset = dataL(P.image_root_dir,P.image_property_dir)
        loader = DataLoader(dataL,batch_size=batch_size,shuffle=True,pin_memory=True ,num_workers=6)
        return loader

    def for_maskG(self,idx):
        '''读取生成数据 image_s, mask
        '''
        if self.maskG == None:
            return None,None
        else:
            return self.maskG()

    def _make_maskG(self,maskG,param,devision):
        G = param[0].cuda()
        bg_direction = param[1]        
        mask_postprocessing = param[2]
        P = copy.deepcopy(param[3])
        if devision == 'real':
            return None
        elif devision == 'split':
            #ori_batch_size = P.batch_size
            P.batch_size = int(P.batch_size/2)
        zs = P.z
        z_noise = P.z_noise

        zs,zs_cls = self._read_train_np(zs)
        maskG = maskG(
            G, bg_direction, P, [], mask_postprocessing,
            zs=zs,zs_cls = zs_cls, z_noise=z_noise).cuda().eval()
        #P.batch_size = ori_batch_size
        return maskG

    def _read_train_np(self,paths):
        '''加载z向量和每一个z向量所对应的类别
        '''
        zs_path = paths
        zs_cls_path = paths.split('.npy')[0]+'_cls.npy'
        zs = torch.from_numpy(np.load(zs_path))
        zs_cls = torch.from_numpy(np.load(zs_cls_path))
        return zs, zs_cls

class CombData(object):
    def __init__(self,dataL,maskG,devision,param):
        '''param结构：
        list： param[0]=G; param[1]=bg_direction, param[2]=mask_postprocessing
        param[3] = param
        '''
        
        self.dataL = self._make_dataL(dataL,param,devision)
        # self.loader = iter(self.dataL)
        self.maskG = self._make_maskG(maskG,param,devision)
        self.devision = devision

    def __getitem__(self,idx):
        '''对split情形的结合选择：
        [真实,合成]
        '''
        _,image_r, clas = self.for_dataL(idx)
        if self.devision == 'real':
            return image_r.cuda(), clas.cuda(), None
        image_s, mask = self.for_maskG(idx)
        if self.devision == 'synthetic':
            return image_s, None ,mask
        image = torch.cat((image_r.cuda(),image_s),0)
        return image, clas.cuda(), mask
    
    def for_dataL(self,idx):
        '''读取真实数据image_r, clas
        '''
        if self.dataL == None:
            return None,None
        else:
            if idx%len(self.dataL) == 0:
                self.loader = iter(self.dataL)
            return next(self.loader)

    def _make_dataL(self,dataL,param,devision):
        P = param[3]
        if devision == 'synthetic':
            return None
        elif devision == 'split':
            batch_size = int(P.batch_size/2)
        else:
            batch_size = P.batch_size
        loader = DataLoader(dataL,batch_size=batch_size,shuffle=True,pin_memory=True ,num_workers=6)
        return loader

    def for_maskG(self,idx):
        '''读取生成数据 image_s, mask
        '''
        if self.maskG == None:
            return None,None
        else:
            return self.maskG()

    def _make_maskG(self,maskG,param,devision):
        G = param[0].cuda()
        bg_direction = param[1]        
        mask_postprocessing = param[2]
        P = copy.deepcopy(param[3])
        if devision == 'real':
            return None
        elif devision == 'split':
            #ori_batch_size = P.batch_size
            P.batch_size = int(P.batch_size/2)
        zs = P.z
        z_noise = P.z_noise
        zs,zs_cls = self._read_train_np(zs)
        maskG = maskG(
            G, bg_direction, P, [], mask_postprocessing,
            zs=zs,zs_cls = zs_cls, z_noise=z_noise).cuda().eval()
        #P.batch_size = ori_batch_size
        return maskG

    def _read_train_np(self,paths):
        '''加载z向量和每一个z向量所对应的类别
        '''
        zs_path = paths
        zs_cls_path = paths.split('.npy')[0]+'_cls.npy'
        zs = torch.from_numpy(np.load(zs_path))
        zs_cls = torch.from_numpy(np.load(zs_cls_path))
        return zs, zs_cls

