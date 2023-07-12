import torch
import os
import cv2
import numpy as np
import argparse
from utils.general import*

def center(x1, y1, x2, y2):
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2

    return center_x, center_y

def tiny_filter(boxes):
    """
    Input:
        boxes: x1, y1, x2, y2, conf
    """
    delta_w = boxes[:, 2] - boxes[:, 0]
    delta_h = boxes[:, 3] - boxes[:, 1]

    keep = (delta_w >=1) & (delta_h >=1)

    return boxes[keep]

if __name__ == '__main__':
    #######################################
    # set up
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="../Datasets/IRSTD/images/", help="path of the image folder/file")
    parser.add_argument("--save_path", type=str, default="mask/", help="path to save results")
    parser.add_argument("--image_save_path", type=str, default="infer/", help="path to save results")
    parser.add_argument('--folder', action='store_true', help='detect images in folder (default:image file)')
    parser.add_argument('--weights', type=str, default="./outputs/demo/last.pt", help="path of the weights")
    parser.add_argument("--iou_thres", type=float, default=0.6, help="iou threshold for detection stage")
    parser.add_argument("--conf_thres", type=float, default=0.2, help="confidence threshold for detection stage")
    parser.add_argument("--topk", type=int, default=5,
                        help="if predict no boxes, select out k region boxes with top confidence")
    parser.add_argument("--expand", type=int, default=8,
                        help="The additonal length of expanded local region for semantic generator")
    parser.add_argument('--fast', action='store_true', help='fast inference')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.image_save_path, exist_ok=True)
    # maskall_path = 'maskall/'
    # block_path = 'block/'
    # os.makedirs(maskall_path, exist_ok=True)
    # os.makedirs(block_path, exist_ok=True)
    #####################################
    # dataset
    args.folder = True
    if args.folder:
        datalist = os.listdir(args.image_path)
    else:
        datalist = [(args.image_path).split('/')[-1]]
    # model
    Model = torch.load(args.weights)
    device = 'cuda'
    Model.to(device)
    Model.eval()

    with torch.no_grad():
        for img_path in datalist:
            if args.folder:
                input = cv2.imread(os.path.join(args.image_path, img_path), 0)
            else:
                input = cv2.imread(args.image_path, 0)

            h, w = input.shape
            img = input[None, None, :]
            # print(img.shape)
            img = np.float32(img) / 255.0

            input = torch.from_numpy(img)  # 1, 1, h, w

            # max_det_num=0 for inference

            detect_output, region_boxes = Model(input.to(device))
            region_boxes = region_boxes.detach()
            # print(detect_output.shape, region_boxes.shape)

            r_boxes = region_boxes[0]
            # print(r_boxes.shape)
            mask_maps = torch.ones((h, w), device=input.device, dtype=torch.bool)  # mask area out of proposed boxes
            mask_maps1 = torch.ones((h, w), device=input.device, dtype=torch.bool)  # mask area out of proposed boxes
            mask_maps2 = torch.ones((h, w), device=input.device, dtype=torch.float32)  # mask area out of proposed boxes
            # output = torch.ones((h, w), device=input.device, dtype=torch.float32)  # mask as the size of target
            output = input
            # print(output.dtype)
            output = output.reshape(h, w)
            # print(output.shape)
            # NMS
            boxes = non_max_suppression(r_boxes, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
            # print(boxes.shape)

            if (region_boxes[:, 4] > args.conf_thres).sum() > 0:
                boxes = non_max_suppression(r_boxes, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
                # filter out boxes that are too small
                boxes = tiny_filter(boxes)
                # print(boxes)
            else:
                boxes = non_max_suppression(r_boxes, conf_thres=args.conf_thres, iou_thres=args.iou_thres)
                # filter out boxes that are too small
                boxes = tiny_filter(boxes)
                boxes = boxes[:args.topk]

            # print(boxes.shape)
            boxes = boxes[:, :4]
            # print(boxes.shape)
            for box in boxes:
                x1, y1, x2, y2 = (box[0]).to(torch.int), (box[1]).to(torch.int), (box[2]).to(torch.int), (
                            box[3]).to(torch.int)

                # print(x1, y1, x2, y2)

                # 跟据图像中像素与中心点的像素差异设置自适应阈值
                block = output[y1:y2, x1:x2]
                # print(block)
                raw, col = block.shape
                # print(raw, col)
                max_block = block.max()
                # print(max_block)
                # 按行展开寻找最大值的位置
                idx_flatted = torch.argmax(block)
                # print(torch.argmax(block), block.reshape(-1)[torch.argmax(block)])
                idx_y = (idx_flatted + 1) // col
                idx_x = (idx_flatted + 1) % col
                print(idx_y, idx_x)
                # 假设目标仅占少于5×5个像素
                block_candidate = block[idx_y-3:idx_y+4, idx_x-3:idx_x+4]
                block_candidate -= max_block
                mean_block_candidate = block_candidate.mean()
                mask_bc = block_candidate.abs() < mean_block_candidate.abs()

                # I_std = block.std(unbiased=False)
                # print('均值：', I_mean, '标准差：', I_std)
                # # th = I_mean + 0.7 * (max_block - I_mean)
                # k = 1
                # th = I_mean + k * I_std

                # diff_raw = block[1:raw, :] - block[0:raw-1, :]
                # diff_col = block[:, 1:] - block[:, :col-1]
                # judge_raw = diff_raw.abs() > 0.05
                # judge_col = diff_col.abs() > 0.05
                # judge = judge_raw[:, :col-1] * judge_col[:raw-1, :]
                # # print(judge.shape)

                # judge_mask = torch.zeros([raw, col], dtype=bool)
                # judge_mask[idx_y-2:idx_y+3, idx_x-2:idx_x+3] = ~mask_bc
                # print(judge_mask.type())

                p = torch.ones([raw, col], dtype=bool)
                p[idx_y-3:idx_y+4, idx_x-3:idx_x+4] = ~mask_bc

                # hello()
                # print(max_block)
                # p = (block > th)
                
                # mask_maps1[y1:y2, x1:x2] = 0
                # mask_maps2[y1:y2, x1:x2] = block[:]
                # print(block.shape)
                # print(mask_maps2[y1:y2, x1:x2].shape)

                # print(output[y1:y2, x1:x2])
                mask_maps[y1:y2, x1:x2] = p
                # print(p)
                # output[y1:y2, x1:x2] = p * output[y1:y2, x1:x2]
                # print(output[y1:y2, x1:x2])

                # # set a fixed target range
                # c_x1, c_y1, c_x2, c_y2 = x - 1, y - 1, x + 1, y + 1
                # # output[y1:y2, x1:x2] = 0.0
                # mask_maps[c_y1:c_y2 + 1, c_x1:c_x2 + 1] = False
                # mask_maps[y, x - 2:x + 3] = False
                # mask_maps[y - 2:y + 3, x] = False
                # output[c_y1:c_y2+1, c_x1:c_x2+1] = 0.0
                # output[y, x-2:x+3] = 0.0
                # output[y-2:y+3, x] = 0.0

            mask_maps = mask_maps.cpu().numpy()
            mask_maps = np.uint8(mask_maps * 255)

            # output = output.cpu().numpy()
            # output = np.uint8(output * 255)
            
            cv2.imwrite(os.path.join(args.save_path, img_path.replace('jpg', 'png')), mask_maps)
            print(f'record: {img_path}')

            # cv2.imwrite(os.path.join(args.image_save_path, img_path.replace('jpg', 'png')), output)

            # cv2.imwrite(os.path.join(maskall_path, img_path.replace('jpg', 'png')), np.uint8(mask_maps1.cpu().numpy() * 255))
            # cv2.imwrite(os.path.join(block_path, img_path.replace('jpg', 'png')), np.uint8((mask_maps2.cpu().numpy()) * 255))


