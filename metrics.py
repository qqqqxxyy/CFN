import numpy as np
import torch
import numpy as np
import cv2
from utils.postprocessing import resize
from tqdm import tqdm
import copy
import ipdb
from utils.postprocessing import *
_CONTOUR_INDEX = 0 #if cv2.__version__.split('.')[0] == '3' else 0
def IoU(mask1, mask2):
    mask1, mask2 = mask1.to(torch.bool), mask2.to(torch.bool)
    intersection = torch.sum(mask1 * (mask1 == mask2), dim=[-1, -2]).squeeze()
    union = torch.sum(mask1 + mask2, dim=[-1, -2]).squeeze()
    return (intersection.to(torch.float) / union).mean().item()


def accuracy(mask1, mask2):
    mask1, mask2 = mask1.to(torch.bool), mask2.to(torch.bool)
    return torch.mean((mask1 == mask2).to(torch.float)).item()


def precision_recall(mask_gt, mask):
    mask_gt, mask = mask_gt.to(torch.bool), mask.to(torch.bool)
    true_positive = torch.sum(mask_gt * (mask_gt == mask), dim=[-1, -2]).squeeze()
    mask_area = torch.sum(mask, dim=[-1, -2]).to(torch.float)
    mask_gt_area = torch.sum(mask_gt, dim=[-1, -2]).to(torch.float)

    precision = true_positive / mask_area
    precision[mask_area == 0.0] = 1.0

    recall = true_positive / mask_gt_area
    recall[mask_gt_area == 0.0] = 1.0

    return precision.item(), recall.item()


def F_score(p, r, betta_sq=0.3):
    f_scores = ((1 + betta_sq) * p * r) / (betta_sq * p + r)
    f_scores[f_scores != f_scores] = 0.0  # handle nans
    return f_scores


def F_max(precisions, recalls, betta_sq=0.3):
    F = F_score(precisions, recalls, betta_sq)
    return F.mean(dim=0).max().item()

def _ct_IoU(IoU,thr=0.5):
    return int(IoU>=thr)



def _mask2loc(gray_map):
    '''
    function: 计算gray_map的最大轮廓坐标       
    input: gray_map: numpy, w*h
        threshold: floar,0-1
    output:estiamted_boxes :[ [x0,y0,x1,y1] ]
    '''
    if torch.is_tensor(gray_map):
        gray_map = gray_map.detach().cpu().numpy().astype('int')
    estimated_boxes = []
    thr_gray_heatmap = (gray_map*255).astype('uint8')
    #gray_map = np.expand_dims((gray_map*255).astype('uint8'),2)
    # _,thr_gray_heatmap = cv2.threshold(
    #     src=gray_map,
    #     thresh=int(threshold*np.max(gray_map)),
    #     maxval=255,
    #     type=cv2.THRESH_BINARY)
    contours= cv2.findContours(
        image=thr_gray_heatmap,
        mode=cv2.RETR_TREE,
        method = cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]
    height ,width = gray_map.shape
    #rx0,ry0,rx1,ry1 = width,height,0,0
    if len(contours)==0:
        x,y,w,h=0,0,1,1
        # xx,yy=x+w,y+h
    else:
        # contours.sort(key = cv2.contourArea, reverse=True)

        contours = [max(contours,key=cv2.contourArea)]
        x,y,w,h = cv2.boundingRect(contours[0])
        # xx,yy=x+w,y+h
        # if len(contours)>1:
        #     x1,y1,w1,h1 = cv2.boundingRect(contours[1])
        #     xx1 ,yy1 = x1+w1,y1+h1
        #     x = min(x1,x)
        #     y = min(y1,y)
        #     xx = max(xx1,xx)
        #     yy = max(yy1,yy)

    rx0,ry0,rx1,ry1 = x,y,x+w,y+h
    # rx0,ry0,rx1,ry1 = x,y,xx,yy
    rx1 = min(rx1,width-1)
    ry1 = min(ry1,height-1)
    estimated_boxes.append([rx0,ry0,rx1,ry1])
    return estimated_boxes


def Localization():
    return 

@torch.no_grad()
def ILSVRC_metrics(segmetation_model, dataloder,n_steps=None, prob_bins=255):

    # avg_values = {}
    # precisions = []
    # recalls = []
    # out_dict = {}
    acc_list = np.array([0])
    n_steps = len(dataloder) if n_steps is None else n_steps
    step = 0
    for step, (img,cls,loc) in tqdm(enumerate(dataloder)):
        img = img.cuda()

        mask_pre = segmetation_model(img)
        cor_gen  = mask2cor(mask_pre.squeeze(),0.5)

        IoU_list = cor2IoU(loc,cor_gen)
        acc_lis = IoU2acc(IoU_list)
        # ipdb.set_trace()
        acc_list+=acc_lis
        if step == n_steps:
            break
    acc_list = acc_list/n_steps
    # outdict = 'accuracy: {}'.format(acc_list)
    return acc_list




@torch.no_grad()
def model_metrics(segmetation_model, dataloder, n_steps=None,
                  stats=(IoU, accuracy, F_max,Localization), prob_bins=255):
    avg_values = {}
    precisions = []
    recalls = []
    out_dict = {}

    n_steps = len(dataloder) if n_steps is None else n_steps
    step = 0
    # print(len(dataloder))
    for step, (img, mask) in tqdm(enumerate(dataloder)):
        img, mask = img.cuda(), mask.cuda()

        if img.shape[-2:] != mask.shape[-2:]:
            mask = resize(mask, img.shape[-2:]) #如果size不一样，回插成原来的尺寸
        prediction = segmetation_model(img)
        
        for metric in stats:
            method = metric.__name__
            if method not in avg_values and metric != F_max:
                avg_values[method] = 0.0
            if method == 'Localization':
                mask,pre = mask.squeeze().cpu().numpy(),prediction.squeeze().cpu().numpy()
                cor_real,cor_pre = _mask2loc(mask>0.5),_mask2loc(pre>0.5)
                IoU = _loc2IoU(cor_real[0],cor_pre[0])
                avg_values[method] += _ct_IoU(IoU)
            elif metric != F_max:
                #如果不是F_max指标，就用在相应记录器中做标准累积
                avg_values[method] += metric(mask, prediction)
            else:
                #iou准则
                p, r = [], []
                splits = 2.0 * prediction.mean(dim=0) if prob_bins is None else \
                    np.arange(0.0, 1.0, 1.0 / prob_bins)

                for split in splits:
                    pr = precision_recall(mask, prediction > split)
                    p.append(pr[0])
                    r.append(pr[1])
                precisions.append(p)
                recalls.append(r)

        step += 1
        if n_steps is not None and step >= n_steps:
            break

    for metric in stats:
        method = metric.__name__
        if metric == F_max:
            out_dict[method] = F_max(torch.tensor(precisions), torch.tensor(recalls))
        else:
            out_dict[method] = avg_values[method] / step

    return out_dict


def msk2IoU_pre(mask_pre_list,mask_orao):
    IoU_list = []
    for mask_pre in mask_pre_list:
        iou = IoU(mask_pre.cpu(),mask_orao)
        IoU_list.append(iou)
    return IoU_list

def mask2IoU(mask_real,mask_pre,pre_thr=None):
    '''对两个mask计算iou
    '''
    if pre_thr == None or pre_thr == -1:
        cor_real,cor_pre = _mask2loc(mask_real>0.5),_mask2loc(mask_pre>0.5)
    else:
        cor_real,cor_pre = _mask2loc(mask_real>0.5),_mask2loc(mask_pre>pre_thr)

    IoU = _loc2IoU(cor_real[0],cor_pre[0])
    return IoU,cor_real,cor_pre

def cal_IoU(mask,mask_pre,thr_list):
    jud_list = []
    cor_true = _mask2loc(mask>0.5)
    for thr in thr_list:
        
        cor_pe = _mask2loc(mask_pre>thr)
        IoU = _loc2IoU(cor_true[0],cor_pe[0])
        jud_list.append(_ct_IoU(IoU))

    return jud_list

def croped_resize(cor_pre_l,image):
    '''将按照image min crop的坐标resize
    回image的尺寸
    '''

    if isinstance(image,list):
        w,h = image
    else:
        h,w = image.shape[1:]
    
    if isinstance(cor_pre_l,list)==False:
        cor_pre_l = [cor_pre_l]
    cor_resized = []

    for cor_pre in cor_pre_l:
        x0,y0,x1,y1 = cor_pre
        if w>h:
            x0 = int(x0+(w-h)/2)
            x1 = int(x1+(w-h)/2)
        elif h>w:
            y0 = int(y0+(h-w)/2)
            y1 = int(y1+(h-w)/2)
        cor_resized.append([x0,y0,x1,y1])
    return cor_resized

def _loc2IoU(cor_gt,cor_gen):
    '''
    function:计算预测bbox和真实bbox之间的IoU
    input: cor_gt,真实坐标，cor_gen预测坐标: [x0,y0,x1,y1]
    '''
    w = min(cor_gt[2],cor_gen[2])-max(cor_gt[0],cor_gen[0])
    if w <= 0 :
        return 0 #the condition which two bbox dont intersect
    h = min(cor_gt[3],cor_gen[3])-max(cor_gt[1],cor_gen[1])
    if h<= 0 :
        return 0
    cross = w*h
    square_2bbox = (cor_gt[2]-cor_gt[0])*(cor_gt[3]-cor_gt[1])+(cor_gen[2]-cor_gen[0])*(cor_gen[3]-cor_gen[1])
    IoU = float(cross)/float((square_2bbox-cross))
    return IoU

def _loc2IoU2(boxA, boxB):
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


def cor2IoU(cor_gen,cor_gt_list):
    '''
    function:计算预测bbox和真实bbox之间的IoU
    input: cor_gt: [x0,y0,x1,y1] or [[x0,y0,x1,y1]], both form is acceptable
        cor_gen预测坐标: [[x0,y0,x1,y1]]
    output: 
    '''
    IoU = []
    #如果不是[[],..,]的形式，就在外面加一个list
    if not isinstance(cor_gt_list[0],list):
        cor_gt_list = [cor_gt_list]
    if not isinstance(cor_gen[0],list):
        cor_gen = [cor_gen]

    for cor in cor_gen: 
        ioU=0
        for cor_gt in cor_gt_list:
            iou_tmp = _loc2IoU(cor_gt,cor)          
            ioU = max(ioU,iou_tmp)
        IoU.append(ioU)

    return IoU

def cor2IoU2(cor_gen,cor_gt_list):
    '''
    和cor2IoU 功能一样，但是按照PsyNet中的test复现的，效果可以超过cor2IoU
    function:计算预测bbox和真实bbox之间的IoU
    input: cor_gt: [x0,y0,x1,y1] or [[x0,y0,x1,y1]], both form is acceptable
        cor_gen预测坐标: [[x0,y0,x1,y1]]
    output: 
    '''
    IoU = []
    #如果不是[[],..,]的形式，就在外面加一个list
    if not isinstance(cor_gt_list[0],list):
        cor_gt_list = [cor_gt_list]
    if not isinstance(cor_gen[0],list):
        cor_gen = [cor_gen]

    for cor in cor_gen: 
        ioU=0
        for cor_gt in cor_gt_list:
            iou_tmp = _loc2IoU2(cor_gt,cor)          
            ioU = max(ioU,iou_tmp)
        IoU.append(ioU)

    return IoU


def mask2cor_rgb(mask,thr):
    '''用thr选择出mask中的前景bbox
    thr可以是list
    _mask2loc,return[[x0,y0,x1,y1]] ,因此需要取首元素
    '''
    heatmap = intensity_to_rgb(np.squeeze(mask), normalize=True).astype('uint8')
    gray_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2GRAY)
    if isinstance(thr,float):
        thr = [thr]
    cor = []
    for tr in thr:
        cor.append(_mask2cor_rgb(gray_heatmap,tr))
    return cor

def _mask2cor_rgb(gray_heatmap,tr):
    th_val = tr * np.max(gray_heatmap)
    _, th_gray_heatmap = cv2.threshold(gray_heatmap, int(th_val), 255, cv2.THRESH_BINARY)
    
    try:
        _, contours, _ = cv2.findContours(th_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv2.findContours(th_gray_heatmap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_box = [x, y, x + w, y + h]
    return estimated_box



def mask2cor(mask,thr):
    '''用thr选择出mask中的前景bbox
    thr可以是list
    _mask2loc,return[[x0,y0,x1,y1]] ,因此需要取首元素
    '''
    if isinstance(thr,float):
        cor = _mask2loc(mask>thr)[0]
    else:
        cor = []
        for tr in thr:
            cor.append(_mask2loc(mask>tr)[0])
    return cor

def maskfilter(mask0,thr):
    if isinstance(thr,float):
        mask = mask0>thr
        cor = _mask2loc(mask)[0]
        x0,y0,x1,y1=cor
        mask[:x0,:]=0
        mask[x1:,:]=0
        mask[:,:y0]=0
        mask[:,y1:]=0
    else:
        mask_list = []
        for tr in thr:
            mask = mask0>tr
            cor = _mask2loc(mask)[0]
            x0,y0,x1,y1=cor
            mask[:y0,:]=0
            mask[y1:,:]=0
            mask[:,:x0]=0
            mask[:,x1:]=0
            mask_list.append(mask)
    return mask_list

def IoU2acc(IoU):
    acc_list = []
    for iou in IoU:
        if iou>=0.5:
            acc_list.append(1)
        else:
            acc_list.append(0)
    return np.array(acc_list)

class ACC_record(object):
    '''inputs:
    acc_list: list with all zero elements, scale matches thr_list
    thr_list: list with elelments of threshold value
    length: total number of counted sample
    outputs:
    average results of iou from thr 0.4 to thr 0.9, avg, miou and detail performance of thr 0.5
    '''
    def __init__(self,acc_list,thr_list):
        self.acc_list_05 = acc_list.astype('float')
        self.acc_list_03 = copy.copy(acc_list)
        self.acc_list_04 = copy.copy(acc_list)
        self.acc_list_06 = copy.copy(acc_list)
        self.acc_list_07 = copy.copy(acc_list)
        self.acc_list_08 = copy.copy(acc_list)
        self.acc_list_09 = copy.copy(acc_list)
        self.thr_list = thr_list
        self.miou = acc_list
        self.length = 0
    def acc_output(self,disp_form):
        length = self.length
        acc_03 = np.max(self.acc_list_03)/length
        acc_03_thr = self.thr_list[np.argmax(self.acc_list_03)]
        acc_04 = np.max(self.acc_list_04)/length
        acc_04_thr = self.thr_list[np.argmax(self.acc_list_04)]
        # ipdb.set_trace()
        acc_05 = np.max(self.acc_list_05)/length
        acc_05_thr = self.thr_list[np.argmax(self.acc_list_05)]
        acc_06 = np.max(self.acc_list_06)/length
        acc_06_thr = self.thr_list[np.argmax(self.acc_list_06)]
        acc_07 = np.max(self.acc_list_07)/length
        acc_07_thr = self.thr_list[np.argmax(self.acc_list_07)]
        acc_08 = np.max(self.acc_list_08)/length
        acc_08_thr = self.thr_list[np.argmax(self.acc_list_08)]
        acc_09 = np.max(self.acc_list_09)/length
        acc_09_thr = self.thr_list[np.argmax(self.acc_list_09)]
        avg = (acc_05+acc_06+acc_07+acc_08+acc_09)/5
        box_v2 = (acc_03+acc_05+acc_07)/3
        miou = np.max(self.miou)/length
        miou_thr = self.thr_list[np.argmax(self.miou)]
        if disp_form==False:
            info_dic={'GT-Loc':[list(self.acc_list_05/length)],'0.3': [acc_03,acc_03_thr], '0.4':[acc_04,acc_04_thr], '0.5':[acc_05,acc_05_thr], \
                '0.6':[acc_06,acc_06_thr], '0.7':[acc_07,acc_07_thr], '0.8':[acc_08,acc_08_thr], '0.9':[acc_09,acc_09_thr], \
                'avg':[avg],'miou':[miou,miou_thr],'box_v2':[box_v2]}
        else :
            info_dic={'GT-Loc':[list(np.around(self.acc_list_05/length,4))],'0.3': [np.around(acc_03,4),round(acc_03_thr,3)], '0.4':[np.around(acc_04,4),round(acc_04_thr,3)],  \
            '0.5':[np.around(acc_05,4),round(acc_05_thr,3)], '0.6':[np.around(acc_06,4),round(acc_06_thr,3)], \
                '0.7':[np.around(acc_07,4) ,round(acc_07_thr,3)], '0.8':[np.around(acc_08,4) ,round(acc_08_thr,3)], \
                '0.9':[np.around(acc_09,4),round(acc_09_thr,3)], 'avg':[np.around(avg,4)],'miou':[np.around(miou,4),round(miou_thr,3)], \
                'box_v2':[np.around(box_v2,4)]  }
        
        return info_dic

    def count(self,IoU_list):
        self.acc_list_03 += self._IoU2acc(IoU_list,0.3)
        self.acc_list_04 += self._IoU2acc(IoU_list,0.4)
        self.acc_list_05 += self._IoU2acc(IoU_list,0.5)
        self.acc_list_06 += self._IoU2acc(IoU_list,0.6)
        self.acc_list_07 += self._IoU2acc(IoU_list,0.7)
        self.acc_list_08 += self._IoU2acc(IoU_list,0.8)
        self.acc_list_09 += self._IoU2acc(IoU_list,0.9)
        self.miou = self.miou+ IoU_list
        self.length+=1
    def _IoU2acc(self,IoU,thr=0.5):
        acc_list = []
        for iou in IoU:
            if iou>=thr:
                acc_list.append(1)
            else:
                acc_list.append(0)
        return np.array(acc_list)

# def max_cor_tmp(cor1,cor2):
#     length = len(cor1)
#     cor_return = []
#     for i in range(length):
#         x01,y01,x11,y11 = cor1[i]
#         x02,y02,x12,y12 = cor2[i]
#         x0,y0 = min(x01,x02),min(y01,y02)
#         x1,y1 = max(x11,x12),max(y11,y12)
#         cor_return.append([x0,y0,x1,y1])
#     return cor_return