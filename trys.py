from torch import nn
from UNet.unet_parts import DoubleConv, Down, Up, OutConv
import ipdb
import torch
from torch.nn import CrossEntropyLoss 
from utils.utils import to_image, acc_counter,loss_counter
import math
from torchvision import transforms
from data import FiledDataset
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from matplotlib import pyplot as plt

import ipdb
from PIL import Image
#制作CUB的Hdf5版本
from utils.prepared_util import ori_filelist, hdf_maker

# train_data_root = '/home/qxy/Desktop/datasets/CUB/data'
# train_property = '/home/qxy/Desktop/datasets/CUB/else_annotation/image_list.json'
# train_hdf_path = '/home/qxy/Desktop/datasets/CUB/train_data.hdf'

train_data_root = '/home/qxy/Desktop/datasets/ILSVRC/data'
train_property = '/home/qxy/Desktop/datasets/ILSVRC/data_annotation/train_list.json'
train_hdf_path = '/mnt/usb/Dataset relate/ILSVRC_hdf5/data.hdf'


Ori_filelist = ori_filelist(train_data_root,train_property)
Hdf_maker = hdf_maker(train_hdf_path,Ori_filelist)
Hdf_maker.make()
print(len(Hdf_maker))