import os
import cv2
import shutil
import torch
import numpy as np
import torchvision
from PIL import Image


def split_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        flag, frame = video_capture.read()
        if not flag:
            break
        # frame_name = '{:05d}.png'.format(idx)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    video_capture.release()
    return frames, fps


def merge(frames, save_path, fps):
    frames = torch.from_numpy(np.asarray(frames))
    torchvision.io.write_video(save_path, frames, fps=fps, video_codec="libx264")


def save_frames(frames, video_fps, save_dir):
    for idx, cur_frame in enumerate(frames):
        Image.fromarray(cur_frame).save(os.path.join((save_dir), str(idx)+'.png'))

# split_num = 4
# split_frames, video_fps = split_video('data/videos/1.mp4')
# split_step = int(len(split_frames) // split_num)
# for i in range(split_num):
#     if i != 3: merge(split_frames[i*split_step:(i+1)*split_step], 'split_{}.mp4'.format(i+1), video_fps)
#     else: merge(split_frames[i*split_step:], 'split_{}.mp4'.format(i+1), video_fps)

frames, video_fps = split_video('./data/videos/1.mp4')
# merge(frames, save_path='tokenflow_24fps.mp4', fps=24)

save_frames(frames, video_fps, './result/frames/building')
