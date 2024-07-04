# from PIL import Image
# import numpy as np
# import os
# from replace import split_video


# def get_avg_mask_area(saved_video_mask_dir):
#     mask_area = 0
#     mask_num = 0
#     for cur_file in os.listdir(saved_video_mask_dir):
#         cur_mask = np.load(os.path.join(saved_video_mask_dir, cur_file), allow_pickle=True)
#         if len(np.unique(cur_mask)) > 1:
#             mask_area += (cur_mask > 0).sum()
#             mask_num += 1
#     return mask_area / mask_num


# avg_mask_area = get_avg_mask_area('data/masks/1')
# print(avg_mask_area)

# for idx in range(199):
#     frame_name, mask_name = '{:05d}.png'.format(idx), '{:05d}.npy'.format(idx)
#     mask = np.load(os.path.join('data/masks/1', mask_name), allow_pickle=True)
#     mask_area = (mask > 0).sum()
#     if mask_area == 0: continue
#     bbx_y, bbx_x = np.nonzero(mask)
#     bbx_top, bbx_bottom, bbx_left, bbx_right = np.min(bbx_y), np.max(bbx_y), np.min(bbx_x), np.max(bbx_x)
#     print('{} {} {} {} {} {}'.format(idx, bbx_top, bbx_bottom, bbx_left, bbx_right, mask_area))
#     if idx > 30: break


# # fps = split_video('new_video.mp4', './debug_video/')

import replicate


model = replicate.models.get('iceclear/stablesr')
model.predict(input_image='./result/frames/tokenflow/0.png')
