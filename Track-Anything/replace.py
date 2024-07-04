import os
import cv2
# from track_anything import parse_augment
from PIL import Image
import shutil
import numpy as np
import math
import copy
import torch
import torchvision
from scipy.ndimage import label
import time
import sys
import argparse
sys.setrecursionlimit(2000)
# from app import model


def split_video(video_path, save_dir, ori_shape):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    if os.path.exists(save_dir): shutil.rmtree(save_dir)
    os.makedirs(save_dir)
    idx = 0
    while True:
        flag, frame = video_capture.read()
        if not flag:
            break
        frame_name = '{:05d}.png'.format(idx)
        idx += 1
        frame = cv2.resize(frame, ori_shape)
        cv2.imwrite(os.path.join(save_dir, frame_name), frame)
    video_capture.release()
    return fps
    
    
def get_avg_mask_area(saved_video_mask_dir):
    mask_area = 0
    mask_num = 0
    for cur_file in os.listdir(saved_video_mask_dir):
        cur_mask = np.load(os.path.join(saved_video_mask_dir, cur_file), allow_pickle=True)
        if len(np.unique(cur_mask)) > 1:
            mask_area += (cur_mask > 0).sum()
            mask_num += 1
    return mask_area / mask_num


def merge_video(res, fps=30, output_path='new_video.mp4'):    
    frames = torch.from_numpy(np.asarray(res))
    # if not os.path.exists(os.path.dirname(output_path)):
    #     os.makedirs(os.path.dirname(output_path))
    torchvision.io.write_video(output_path, frames, fps=fps, video_codec="libx264")


def find_area(cur_i, cur_j, bbx_top, bbx_bottom, bbx_left, bbx_right, block_area, flag_idx, mask, flag):
    block_area += 1
    shape_x, shape_y = mask.shape[:2]
    flag[cur_i][cur_j] = flag_idx
    if cur_i > 0 and mask[cur_i-1][cur_j] == 1 and flag[cur_i-1][cur_j] == 0:
        bbx_top = min(cur_i-1, bbx_top)
        bbx_top_, bbx_bottom_, bbx_left_, bbx_right_, block_area_, new_flag = find_area(cur_i-1, cur_j, bbx_top, bbx_bottom, bbx_left, bbx_right, block_area, flag_idx, mask, flag)
        bbx_top, bbx_left = min(bbx_top, bbx_top_), min(bbx_left, bbx_left_)
        bbx_bottom, bbx_right, block_area = max(bbx_bottom, bbx_bottom_), max(bbx_right, bbx_right_), max(block_area, block_area_)
        flag[new_flag==flag_idx] = flag_idx
    if cur_i < shape_x-1 and mask[cur_i+1][cur_j] == 1 and flag[cur_i+1][cur_j] == 0:
        bbx_bottom = max(cur_i+1, bbx_bottom)
        bbx_top_, bbx_bottom_, bbx_left_, bbx_right_, block_area_, new_flag  = find_area(cur_i+1, cur_j, bbx_top, bbx_bottom, bbx_left, bbx_right, block_area, flag_idx, mask, flag)
        bbx_top, bbx_left = min(bbx_top, bbx_top_), min(bbx_left, bbx_left_)
        bbx_bottom, bbx_right, block_area = max(bbx_bottom, bbx_bottom_), max(bbx_right, bbx_right_), max(block_area, block_area_)
        flag[new_flag==flag_idx] = flag_idx
    if cur_j > 0 and mask[cur_i][cur_j-1] == 1 and flag[cur_i][cur_j-1] == 0:
        bbx_left = min(cur_j-1, bbx_left)
        bbx_top_, bbx_bottom_, bbx_left_, bbx_right_, block_area_, new_flag  = find_area(cur_i, cur_j-1, bbx_top, bbx_bottom, bbx_left, bbx_right, block_area, flag_idx, mask, flag)
        bbx_top, bbx_left = min(bbx_top, bbx_top_), min(bbx_left, bbx_left_)
        bbx_bottom, bbx_right, block_area = max(bbx_bottom, bbx_bottom_), max(bbx_right, bbx_right_), max(block_area, block_area_)
        flag[new_flag==flag_idx] = flag_idx
    if cur_j < shape_y-1 and mask[cur_i][cur_j+1] == 1 and flag[cur_i][cur_j+1] == 0:
        bbx_right = max(cur_j+1, bbx_right)
        bbx_top_, bbx_bottom_, bbx_left_, bbx_right_, block_area_, new_flag  = find_area(cur_i, cur_j+1, bbx_top, bbx_bottom, bbx_left, bbx_right, block_area, flag_idx, mask, flag)
        bbx_top, bbx_left = min(bbx_top, bbx_top_), min(bbx_left, bbx_left_)
        bbx_bottom, bbx_right, block_area = max(bbx_bottom, bbx_bottom_), max(bbx_right, bbx_right_), max(block_area, block_area_)
        flag[new_flag==flag_idx] = flag_idx
    return bbx_top, bbx_bottom, bbx_left, bbx_right, block_area, flag
    

def filter_mask(mask):
    labeled, num_features = label(mask)
    if num_features <= 1: return mask
    
    areas, locs = [], []
    for cur_label in range(1, num_features+1): 
        areas.append((labeled==cur_label).sum())
        x_idx, y_idx = np.where(labeled==cur_label)
        bbx_top, bbx_bottom, bbx_left, bbx_right = min(x_idx), max(x_idx), min(y_idx), max(y_idx)
        locs.append([(bbx_top+bbx_bottom)/2, (bbx_left+bbx_right)/2])
    
    # flag = np.zeros_like(mask, dtype=float)
    # areas, locs = [], []
    # flag_idx = 1
    
    # for i in range(mask.shape[0]):
    #     for j in range(mask.shape[1]):
    #         if mask[i][j] == 0: flag[i][j] = -1
    #         if flag[i][j] != 0: continue
    #         bbx_top, bbx_bottom, bbx_left, bbx_right = i, i, j, j
    #         block_area = 0
    #         bbx_top, bbx_bottom, bbx_left, bbx_right, block_area, new_flag = find_area(i, j, bbx_top, bbx_bottom, bbx_left, bbx_right, block_area, flag_idx, mask, flag)
    #         areas.append(block_area)
    #         locs.append([(bbx_top+bbx_bottom)/2, (bbx_left+bbx_right)/2])
    #         flag[new_flag==flag_idx] = flag_idx
    #         flag_idx += 1
    
    if len(areas) > 1:
        max_idx = np.argmax(np.array(areas))
        max_center = locs[max_idx]
        false_idxes = []
        for idx in range(len(areas)):
            if idx == max_idx: continue
            cur_center = locs[idx]
            cur_dis = math.sqrt((cur_center[0]-max_center[0])**2 + (cur_center[1]-max_center[1])**2)
            if cur_dis > 5:
                false_idxes.append(idx+1)
    
        for cur_idx in false_idxes: mask[labeled == cur_idx] = 0
    return mask


def replace_with_object(object_img_path, object_mask_path, saved_video_frame_dir, saved_video_mask_dir, fps=30):
    object_image = Image.open(object_img_path).convert("RGB")
    object_mask = Image.open(object_mask_path)
    object_image, object_mask = np.array(object_image, dtype=np.uint8), np.array(object_mask, dtype=np.uint8)
    object_image_h, object_image_l = object_image.shape[:2]
    object_mask_transparency = object_mask[:, :, 3]
    object_mask_transparency[object_mask_transparency <= 100] = 0
    object_mask_transparency[object_mask_transparency != 0] = 1
    
    video_len, mask_len = len(os.listdir(saved_video_frame_dir)), len(os.listdir(saved_video_mask_dir))
    # assert video_len == mask_len
    
    res = []
    avg_mask_area = get_avg_mask_area(saved_video_mask_dir)
    
    for idx in range(video_len):
        frame_name, mask_name = '{:05d}.png'.format(idx), '{:05d}.npy'.format(idx)
        frame = Image.open(os.path.join(saved_video_frame_dir, frame_name)).convert("RGB")
        frame = np.array(frame, dtype=np.uint8)
        mask = np.load(os.path.join(saved_video_mask_dir, mask_name), allow_pickle=True)
        mask_area = (mask > 0).sum()
        if mask_area <= avg_mask_area*0.8: 
            res.append(frame)
            # new_mask = np.zeros_like(mask)
            # np.save(os.path.join(save_new_mask_dir, '{:05d}'.format(idx)), new_mask)
            continue
        
        mask = filter_mask(mask)
        
        bbx_y, bbx_x = np.nonzero(mask)
        bbx_top, bbx_bottom, bbx_left, bbx_right = np.min(bbx_y), np.max(bbx_y), np.min(bbx_x), np.max(bbx_x)
        times_y, times_x = (bbx_bottom-bbx_top) / object_image_h, (bbx_right-bbx_left) / object_image_l
        total_times = max(times_y, times_x)
        cur_image_h, cur_image_l = math.ceil(object_image_h*total_times), math.ceil(object_image_l*total_times)
        copy_image = copy.deepcopy(object_image)
        copy_mask = copy.deepcopy(object_mask_transparency)
        copy_image = cv2.resize(copy_image, (cur_image_l, cur_image_h))
        copy_mask = cv2.resize(copy_mask, (cur_image_l, cur_image_h), cv2.INTER_NEAREST)

        center_y, center_x = int((bbx_bottom+bbx_top)/2), int((bbx_left+bbx_right)/2)
        replace_top, replace_left = center_y-int(cur_image_h/2), center_x-int(cur_image_l/2)
        replace_bottom, replace_right = replace_top + cur_image_h, replace_left+cur_image_l
        if replace_top < 0:
            copy_image = copy_image[-replace_top:, :, :]
            copy_mask = copy_mask[-replace_top:, :]
            replace_top = 0
        if replace_left < 0:
            copy_image = copy_image[:, -replace_left:, :]
            copy_mask = copy_mask[:, -replace_left:]
            replace_left = 0
        if replace_bottom > frame.shape[0]:
            copy_image = copy_image[:frame.shape[0], :, :]
            copy_mask = copy_mask[:frame.shape[0], :]
            replace_bottom = frame.shape[0]
        if replace_right > frame.shape[1]:
            copy_image = copy_image[:, :frame.shape[1], :]
            copy_mask = copy_mask[:, :frame.shape[1]]
            replace_right = frame.shape[1]
        rec_frame = frame[replace_top:replace_bottom, replace_left:replace_right, :]
        copy_image[copy_mask==0] = rec_frame[copy_mask==0]
        frame[replace_top:replace_bottom, replace_left:replace_right, :] = copy_image
        
        # new_mask = np.zeros((frame.shape[0], frame.shape[1]))
        # local_mask = np.zeros((replace_bottom-replace_top, replace_right-replace_left))
        # local_mask[copy_mask==1] = 1
        # new_mask[replace_top:replace_bottom, replace_left:replace_right] = local_mask
        # np.save(os.path.join(save_new_mask_dir, '{:05d}'.format(idx)), new_mask)
        res.append(frame)

    merge_video(res, fps)
    
    
def replace_with_origin_mask(object_img_path, object_mask_path, saved_video_frame_dir, saved_video_mask_dir, fps):
    object_image = Image.open(object_img_path).convert("RGB")
    object_mask = Image.open(object_mask_path)
    object_image, object_mask = np.array(object_image, dtype=np.uint8), np.array(object_mask, dtype=np.uint8)
    object_image_h, object_image_l = object_image.shape[:2]
    object_mask_transparency = object_mask[:, :, 3]
    object_mask_transparency[object_mask_transparency <= 100] = 0
    object_mask_transparency[object_mask_transparency != 0] = 1
    
    video_len, mask_len = len(os.listdir(saved_video_frame_dir)), len(os.listdir(saved_video_mask_dir))
    # assert video_len == mask_len
    
    res = []
    
    for idx in range(video_len):
        frame_name, mask_name = '{:05d}.png'.format(idx), '{:05d}.npy'.format(idx)
        frame = Image.open(os.path.join(saved_video_frame_dir, frame_name)).convert("RGB")
        frame = np.array(frame, dtype=np.uint8)
        mask = np.load(os.path.join(saved_video_mask_dir, mask_name), allow_pickle=True)
        
        if mask.sum() == 0: 
            res.append(frame)
            continue
        
        bbx_y, bbx_x = np.nonzero(mask)
        bbx_top, bbx_bottom, bbx_left, bbx_right = np.min(bbx_y), np.max(bbx_y), np.min(bbx_x), np.max(bbx_x)

        cur_image_h, cur_image_l = math.ceil(bbx_bottom-bbx_top), math.ceil(bbx_right-bbx_left)

        copy_image = copy.deepcopy(object_image)
        copy_image = cv2.resize(copy_image, (cur_image_l, cur_image_h))

        rec_frame = copy.deepcopy(frame)
        frame[bbx_top:bbx_bottom, bbx_left:bbx_right, :] = copy_image
        frame[mask==0] = rec_frame[mask==0]
        res.append(frame)
    print(len(res))
    merge_video(res, fps)


parser = argparse.ArgumentParser()
parser.add_argument('--ref_img', type=str, default='./data/ref_imgs/4.png')
parser.add_argument('--ref_mask', type=str, default='./data/ref_masks/4.png')
opt = parser.parse_args()
if not os.path.exists(opt.ref_img):
    print('Please check the path of reference image')
    exit(0)
if not os.path.exists(opt.ref_mask):
    print('Please check the path of reference mask. Note that the mask can be downloaded from aliyun API.')
    exit(0)

f = open('info.txt', 'r')
video_name = f.readline().strip().split('.')[0]
words = f.readline().strip().split()
H, W = int(words[0]), int(words[1])
fps = split_video('transferred_video.mp4', 'data/tmp_frames', (W, H))
start_time = time.time()
replace_with_object('data/ref_imgs/4.png', 'data/ref_masks/4.png', 'data/tmp_frames', 'result/mask/'+video_name, fps)
# replace_with_origin_mask('data/ref_imgs/4.png', 'data/ref_masks/4.png', 'data/tmp_frames', 'data/new_masks/1', fps)
end_time = time.time()
print(end_time-start_time)
