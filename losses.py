'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2022-11-27 20:38:40
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
LastEditTime: 2022-11-28 15:51:54
FilePath: \20210123e:\深兰学习\杜老师课件\my\loss-my.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn

def focal_loss(predict, target):
    alpha = 2
    beta = 4
    positive_mask = target.eq(1).float()
    positive_loss = torch.pow((1 - predict), alpha) * torch.log(predict) * positive_mask
    negative_loss = torch.pow((1 - target), beta) * torch.pow(predict, alpha) * torch.log(1 - predict)
    # print(negative_loss)
    N = positive_mask.sum()
    if N == 0: N = 1
    return -1 / N * (positive_loss + negative_loss).sum()

def smooth_l1_loss_modify(predict, target, mask, sigma=3):
    # predict: bx2x96x128
    # target : bx2x96x128
    # mask   : bx2x96x128
    num_object = mask.sum().item() / mask.size(1)
    sigma2 = sigma * sigma
    diff = predict[mask] - target[mask]
    diff_abs = diff.abs()
    near = (diff_abs < 1 / sigma2).float()
    far = 1 - near
    return (near * 0.5 * sigma2 * torch.pow(diff, 2) + far * (diff_abs - 0.5 / sigma2)).sum() / num_object
    
def l2_loss_modify(predict, target, mask):
    # predict: bx2x96x128
    # target : bx2x96x128
    # mask   : bx2x96x128
    num_object = mask.sum().item() / mask.size(1)
    if num_object == 0 : num_object = 1
    masked_predict = predict[mask]
    masked_target = target[mask]
    return torch.pow(masked_predict - masked_target, 2).sum() / num_object

def l1_loss_modify(predict, target, mask):
    # predict: bx2x96x128
    # target : bx2x96x128
    # mask   : bx2x96x128
    num_object = mask.sum().item() / mask.size(1)
    if num_object == 0 : num_object = 1
    masked_predict = predict[mask]
    masked_target = target[mask]
    return torch.abs(masked_predict - masked_target).sum() / num_object

class  GIoULoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,A,B):
        num_bbox = A.size(0)
        if num_bbox == 0:
            return 0
            
        ax, ay, ar, ab = A[:, 0], A[:, 1], A[:, 2], A[:, 3]
        bx, by, br, bb = B[:, 0], B[:, 1], B[:, 2], B[:, 3]
        xmax = torch.min(ar, br)
        ymax = torch.min(ab, bb)
        xmin = torch.max(ax, bx)
        ymin = torch.max(ay, by)
        cross_width = (xmax - xmin + 1).clamp(0)
        cross_height = (ymax - ymin + 1).clamp(0)
        cross = cross_width * cross_height
        union = (ar - ax + 1) * (ab - ay + 1) + (br - bx + 1) * (bb - by + 1) - cross
        iou = cross / union
        cxmin = torch.min(ax, bx)
        cymin = torch.min(ay, by)
        cxmax = torch.max(ar, br)
        cymax = torch.max(ab, bb)
        c = (cxmax - cxmin + 1) * (cymax - cymin + 1)
        return (1 - (iou - (c - union) / c)).sum() / num_bbox / 2