import torch
from torch import nn
from torch._C import dtype
import torch.nn.functional as F
import numpy as np
import random
import os

from utils.general import*
from .embedding import*
from .backbone import semantic_estab

# filter out extremely tiny bounding boxes
def tiny_filter(boxes):
    """
    Input:
        boxes: x1, y1, x2, y2, conf
    """
    delta_w = boxes[:, 2] - boxes[:, 0]
    delta_h = boxes[:, 3] - boxes[:, 1]

    keep = (delta_w >=1) & (delta_h >=1)

    return boxes[keep]

def RandomSelectNegRegion(x, num, smin=5, smax=8):
    C, H, W = x.shape
    #random select number of negative boxes
    #num = random.randint(1, num)
    num = num
    neg_boxes = []
    for n in range(num):
        cx = random.randint(smax, W-smax)
        cy = random.randint(smax, H-smax)
        rw = random.randint(smin, smax)
        rh = random.randint(smin, smax)

        neg_boxes.append(torch.tensor([cx-rw, cy-rh, cx+rw, cy+rh, 0.5], dtype=torch.float))
    if num == 0:
        neg_boxes = None
    else:
        neg_boxes = torch.stack(neg_boxes, dim=0)
    return neg_boxes

class box(nn.Module):
    def __init__(self, region_module):
        super().__init__()
        self.region_module = region_module


    def forward(self, x, max_det_num=10, conf_thres=0.2, iou_thres=0.4, expand=10, topk=5, fast=False):
        b, c, h, w = x.shape

        detect_output, region_boxes = self.region_module(x)
        region_boxes = region_boxes.detach()

        mask_maps = torch.ones((b, h, w), device=x.device, dtype=torch.bool)  # mask area out of proposed boxes

        target_boxes = []
        max_words_num = 0

        region_boxes_exists = True  # using at inference time to judge whether detector find targets
        max_words_num = 0
        for i in range(b):
            r_boxes = region_boxes[i]
            # NMS
            boxes = non_max_suppression(r_boxes, conf_thres=conf_thres, iou_thres=iou_thres)

            if (r_boxes[:, 4] > conf_thres).sum() > 0:
                boxes = non_max_suppression(r_boxes, conf_thres=conf_thres, iou_thres=iou_thres)
                # filter out boxes that are too small
                boxes = tiny_filter(boxes)

            else:
                boxes = non_max_suppression(r_boxes, conf_thres=0.0, iou_thres=iou_thres)
                # filter out boxes that are too small
                boxes = tiny_filter(boxes)
                boxes = boxes[:topk]

            # At inference time, if detector proposed no region boxes, record and jump out.
            if (self.training == False) & (len(boxes) == 0):
                region_boxes_exists = False
                break

            if len(boxes) < max_det_num:
                # Select negative region
                neg_boxes = RandomSelectNegRegion(x[i], max_det_num - len(boxes), len(boxes))
                # combine
                if neg_boxes is not None:
                    neg_boxes = neg_boxes.to(boxes.device)
                    boxes = torch.cat([boxes, neg_boxes], dim=0)
                # Keep no overlap
                boxes = non_max_suppression(boxes, conf_thres=0.0, iou_thres=iou_thres)

            elif self.training:
                boxes = boxes[:max_det_num]

            boxes = boxes[:, :4]

            target_box = []

            mask_map = mask_maps[i]
            for box in boxes:
                x1, y1, x2, y2 = (box[0] + 0.5).to(torch.int), (box[1] + 0.5).to(torch.int), (box[2] + 0.5).to(
                    torch.int), (box[3] + 0.5).to(torch.int)
                target_box.append([x1.item(), y1.item(), x2.item(), y2.item()])

                mask_map[y1:y2, x1:x2] = False

            target_boxes.append(target_box)





        return (detect_output, target_boxes) if self.training else (seg_output, mask_maps, region_boxes, target_boxes)