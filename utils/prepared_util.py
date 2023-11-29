import json
import os
from unittest import expectedFailure
import ipdb
import sys
import numpy as np
from tqdm import tqdm
import random
from PIL import Image
from utils.io_util import dataset_file_load
import h5py
import ipdb
from torchvision import transforms

#defining HDF5format
class ori_filelist(object):
    def __init__(self,images_root,image_prop,tsfm=None,mask=False):
        self.load_tool = dataset_file_load()
        #read image path list
        self.image_path_list = self.load_tool.read_path_list(image_prop,images_root,mask)
        self.relative_path = self.load_tool._read_half_path(image_prop,mask)
    def path_list(self):
        return self.image_path_list,self.relative_path


class hdf_maker(object):
    def __init__(self,store_path,read_path_list):
        self.read_list, self.relative_path = read_path_list.path_list()
        self.store_path = store_path
        

    def make(self,compress=True):
        length = len(self.read_list)
        lengths = 0
        # length = 50
        with h5py.File(self.store_path,'w') as hdf:
            for idx in tqdm(range(length)):
                current_img_path = self.read_list[idx]
                img_data = self.read_img_data(current_img_path)
                img_name = self.relative_path[idx]                
                if compress == True: 
                    hdf.create_dataset(img_name,data=img_data,compression="gzip", compression_opts=9)
                else:       
                    hdf.create_dataset(img_name,data=img_data)
                lengths+=1
            
            hdf.create_dataset('_Num',data=lengths)

    def adding(self,ori_filelist,compress=True):
        read_list,relative_path = ori_filelist.path_list()


        length = len(read_list)
        lengths = 0
        # length = 50
        with h5py.File(self.store_path,'r+') as hdf:
            for idx in tqdm(range(length)):
                current_img_path = read_list[idx]
                img_data = self.read_img_data(current_img_path)
                img_name = relative_path[idx]     
                if compress == True: 
                    hdf.create_dataset(img_name,data=img_data,compression="gzip", compression_opts=9)
                else:       
                # try:
                    hdf.create_dataset(img_name,data=img_data)
                # except:
                #     del hdf[img_name]
                    # hdf.create_dataset(img_name,data=img_data)
                lengths+=1
            curlen = self.__len__()
            del hdf['_Num']
            hdf.create_dataset('_Num',data=lengths+curlen)

    def read_img_data(self,path):
        image = Image.open(path)
        return image

    def fetch(self,img_name=None,idx=None):
        if idx != None:
            img_name = self.relative_path[idx]
        with h5py.File(self.store_path,'r') as hdf:
            data = np.array(hdf.get(img_name))

        return data

    def __len__(self):
        with h5py.File(self.store_path,'r') as hdf:
            return np.array(hdf.get('_Num'))

class hdf_reader(object):
    pass

# #Find the paths of paths.json, based on the this file's abs root
# PATH_FILE = os.path.abspath(os.path.split(__file__)[0]+'/../../weight/paths.json')

# def load_path_file():
#     with open(PATH_FILE,'r') as f:
#         path_file = json.load(f)
#     return path_file


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

class make_property():
    def __init__(self,proper_path,store_path,img_suffix='.jpg'):
        '''proper_path:读取信息的来源
        store_path:保存的路径
        '''
        self.proper_path = proper_path
        self.store_path = store_path
        self.property_list = []
        self.class_list ,self.path_list,self.name_list = \
            self._make_property_list(img_suffix)


    def make_property(self):
        length = len(self.class_list)
        with open(self.store_path,'w') as f:
            f.write('[\n')
            #mini trial size
            for idx in tqdm(range(length)):
                pro_dict = self._make_pro_dict(idx)
                json.dump(pro_dict,f)
                f.write(',\n')
            f.write(']')

    def _make_pro_dict(self,idx):
        No = idx + 1
        name = self.name_list[idx]
        clas = self.class_list[idx]
        path = self.path_list[idx]
        pro_dict = {"No":No,"name":name,"class":clas,"path":path}
        return pro_dict

    def _make_property_list(self,img_suffix):
        #idx 从0开始，No从1开始
        class_list , path_list, name_list = \
            [],[],[]
        with open(self.proper_path,'r') as pros:
            f = pros.readlines()
            # if self.length != None:
            #     f = f[:self.length]
            for pro in f:
                path = pro.split(' ')[0]
                clas = int(pro.split(' ')[-1].split('\n')[0])+1
                name  = path.split('/')[-1].split(img_suffix)[0]
                class_list.append(clas)
                path_list.append(path)
                name_list.append(name)
        return class_list,path_list,name_list

class make_test_property(make_property):
    def __init__(self,proper_path,store_path,sizebox_path,length=None):
        self.length=length
        super(make_test_property, self).__init__(proper_path,store_path)
        self.sizebox_path = sizebox_path
        self.size_list ,self.bbox_list = self._make_sizebox_list()

    def _make_sizebox_list(self):
        length = len(self.class_list)
        size_list = []
        bbox_list = []
        with open(self.sizebox_path,'r') as f:
            sizebox_dict = json.load(f)
            for idx in range(length):
                No = idx+1
                size_list.append(sizebox_dict[str(No)]['size'])
                bbox_list.append(sizebox_dict[str(No)]['bbox'])
        return size_list,bbox_list

    def _make_pro_dict(self,idx):
        No = idx+1
        name = self.name_list[idx]
        clas = self.class_list[idx]
        path = self.path_list[idx]
        size = self.size_list[idx]
        bbox = self.bbox_list[idx]
        pro_dict = {"No":No,"name":name,"bbox":bbox,"class":clas,"size":size,"path":path}
        return pro_dict


class Car_train(make_property):

    def _make_pro_dict(self,idx):
        No = idx + 1
        name = self.name_list[idx]
        clas = self.class_list[idx]
        path = self.path_list[idx]
        size = [self.size_list[idx]]
        pro_dict = {"No":No,"name":name,"class":clas,"size":size,"path":path}
        return pro_dict

    def make_size_list(self,root_path):
        length = len(self.class_list)
        size_list = []
        for i in tqdm(range(length)):
            dirpath = self.path_list[i]
            paths = os.path.join(root_path,dirpath)
            size_list.append(list(Image.open(paths).size))
        self.size_list = size_list
        print('size list make done')

class Car_test(Car_train):
    '''input: info_read.json for getting the information with form need adjust, 
        info_resotre.json for storing write information
    '''
    def __init__(self,proper_path,store_path,img_suffix='.jpg'):
        '''proper_path:读取信息的来源
        store_path:保存的路径
        '''
        self.proper_path = proper_path[0]
        self.proper_path2 = proper_path[1]
        self.store_path = store_path
        self.property_list = []

        self.path_list,self.class_list,self.name_list,self.bbox_list,self.size_list = \
            self._make_property_list(img_suffix)

    def _make_property_list(self, img_suffix):
        class_list , path_list, name_list ,bbox_list ,size_list= \
            [],[],[],[],[]
        with open(self.proper_path,'r') as f:
            property_list = json.load(f)
            for i in range(3333):
                proper = property_list[str(i+1)]
                bbox_list.append(proper['bbox'])
                size_list.append([proper['size']])
        with open(self.proper_path2,'r') as pros:
            f = pros.readlines()
            for pro in f:
                path = pro.split(' ')[0]
                clas = int(pro.split(' ')[-1].split('\n')[0])+1
                name  = path.split('/')[-1].split(img_suffix)[0]
                class_list.append(clas)
                path_list.append(path)
                name_list.append(name)
        return path_list,class_list,name_list,bbox_list,size_list        

    def _make_pro_dict(self,idx):
        No = idx + 1
        name = self.name_list[idx]
        clas = self.class_list[idx]
        path = self.path_list[idx]
        bbox = self.bbox_list[idx]
        size = self.size_list[idx]
        pro_dict = {"No":No,"name":name,"bbox":bbox,"class":clas,"size":size,"path":path}
        return pro_dict

def Car_val(proper_path,store_path,length):
    val_list = []
    with open(proper_path,'r') as f:
        proper_lis = json.load(f)

        randlist = [x for x in range(len(proper_lis))]
        random.shuffle(randlist)
        randlist = randlist[:length]
        randlist.sort()

        for idx in randlist:
            val_list.append(proper_lis[idx])

        with open(store_path,'w') as f:
            f.write('[\n')
            for proper in val_list:
                json.dump(proper,f)
                f.write(',\n')
            f.write(']')

    
def make_property(self):
    length = len(self.class_list)
    with open(self.store_path,'w') as f:
        f.write('[\n')
        #mini trial size
        for idx in tqdm(range(length)):
            pro_dict = self._make_pro_dict(idx)
            json.dump(pro_dict,f)
            f.write(',\n')
        f.write(']')
