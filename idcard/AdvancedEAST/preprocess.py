import os
import random

import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw

from AdvancedEAST import cfg
from AdvancedEAST.label import shrink

def batch_reorder_vertexes(xy_list_array):
    reorder_xy_list_array = np.zeros_like(xy_list_array)
    for xy_list, i in zip(xy_list_array, range(len(xy_list_array))):
        reorder_xy_list_array[i] = reorder_vertexes(xy_list)
    return reorder_xy_list_array


def reorder_vertexes(xy_list):
    reorder_xy_list = np.zeros_like(xy_list)
    #首先找到左上方的点
    #用x最小值确定第一个点，如果有两个点的x值相同，以最小的y为准
    ordered = np.argsort(xy_list, axis=0) #各点x升序排序
    x_min1_index = ordered[0, 0]
    x_min2_index = ordered[1, 0] #取前两个最小的x值比较大小 相同再比较y
    if xy_list[x_min1_index, 0] == xy_list[x_min2_index, 0]:
        if xy_list[x_min1_index, 1] <= xy_list[x_min2_index, 1]:
            reorder_xy_list[0] = xy_list[x_min1_index]
            first_v = x_min1_index
        else:
            reorder_xy_list[0] = xy_list[x_min2_index]
            first_v = x_min2_index
    else:
        reorder_xy_list[0] = xy_list[x_min1_index]
        first_v = x_min1_index
    others = list(range(4)) # 记录剩余点
    others.remove(first_v)  # 去掉已经确定好的点
    k = np.zeros((len(others),))
    # 根据第一个点与其他三点连线的斜率 确定第三个点 斜率居中的即为第三个点
    for index, i in zip(others, range(len(others))):
        k[i] = (xy_list[index, 1] - xy_list[first_v, 1]) \
                    / (xy_list[index, 0] - xy_list[first_v, 0] + cfg.epsilon)
        # 斜率k=（y_i-y_1）/（x_i-x_1）
    k_mid = np.argsort(k)[1]
    third_v = others[k_mid]
    reorder_xy_list[2] = xy_list[third_v]
    others.remove(third_v)
    b_mid = xy_list[first_v, 1] - k[k_mid] * xy_list[first_v, 0]
    second_v, fourth_v = 0, 0
    # 确定第二个点和第四个点
    for index, i in zip(others, range(len(others))):
        # △y = y - (k * x + b).
        delta_y = xy_list[index, 1] - (k[k_mid] * xy_list[index, 0] + b_mid)
        if delta_y > 0:
            second_v = index
        else:
            fourth_v = index
    reorder_xy_list[1] = xy_list[second_v]
    reorder_xy_list[3] = xy_list[fourth_v]
    # 比较两条对角线斜率 保证四个点的顺序正确
    k13 = k[k_mid]
    k24 = (xy_list[second_v, 1] - xy_list[fourth_v, 1]) / (
                xy_list[second_v, 0] - xy_list[fourth_v, 0] + cfg.epsilon)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list


def resize_image(im, max_img_size=cfg.max_train_img_size):
    im_width = np.minimum(im.width, max_img_size)
    if im_width == max_img_size < im.width:
        im_height = int((im_width / im.width) * im.height)
    else:
        im_height = im.height
    o_height = np.minimum(im_height, max_img_size)
    if o_height == max_img_size < im_height:
        o_width = int((o_height / im_height) * im_width)
    else:
        o_width = im_width
    d_wight = o_width - (o_width % 32)
    d_height = o_height - (o_height % 32)
    return d_wight, d_height


def preprocess():
    data_dir = cfg.data_dir
    origin_image_dir = os.path.join(data_dir, cfg.origin_image_dir_name)
    origin_txt_dir = os.path.join(data_dir, cfg.origin_txt_dir_name)
    train_image_dir = os.path.join(data_dir, cfg.train_image_dir_name)
    train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)
    if not os.path.exists(train_image_dir):
        os.mkdir(train_image_dir)
    if not os.path.exists(train_label_dir):
        os.mkdir(train_label_dir)
    draw_gt_quad = cfg.draw_gt_quad
    show_gt_image_dir = os.path.join(data_dir, cfg.show_gt_image_dir_name)
    if not os.path.exists(show_gt_image_dir):
        os.mkdir(show_gt_image_dir)
    show_act_image_dir = os.path.join(cfg.data_dir, cfg.show_act_image_dir_name)
    if not os.path.exists(show_act_image_dir):
        os.mkdir(show_act_image_dir)

    o_img_list = os.listdir(origin_image_dir)
    print("Found %d origin images." % len(o_img_list))
    train_val_set = []
    for o_img_fname, _ in zip(o_img_list, tqdm(range(len(o_img_list)))):
        o_img_name, ext = os.path.splitext(o_img_fname)
        img_path = os.path.join(origin_image_dir, o_img_fname)
        with Image.open(img_path) as im:
            # d_wight, d_height = resize_image(im)
            d_width, d_height = cfg.max_train_img_size, cfg.max_train_img_size
            scale_ratio_w = d_width / im.width
            scale_ratio_h = d_height / im.height
            im = im.resize((d_width, d_height), Image.NEAREST).convert('RGB')
            show_gt_im = im.copy()
            # draw on the img.
            draw = ImageDraw.Draw(show_gt_im)
            txt_path = os.path.join(origin_txt_dir, o_img_name + ".txt")
            with open(txt_path, 'r') as f:
                anno_list = f.readlines()
            xy_list_array = np.zeros((len(anno_list), 4, 2))
            for anno, i in zip(anno_list, range(len(anno_list))):
                anno_colums = anno.strip().split(',')
                anno_array = np.array(anno_colums)
                xy_list = np.reshape(anno_array[:8].astype(float), (4, 2))
                xy_list[:, 0] = xy_list[:, 0] * scale_ratio_w
                xy_list[:, 1] = xy_list[:, 1] * scale_ratio_h
                xy_list = reorder_vertexes(xy_list)
                xy_list_array[i] = xy_list
                _, shrink_xy_list, _ = shrink(xy_list,cfg.shrink_ratio)
                shrink_1, _, long_edge = shrink(xy_list, cfg.shrink_side_ratio)
                if draw_gt_quad:
                    draw.line([tuple(xy_list[0]), tuple(xy_list[1]),
                               tuple(xy_list[2]), tuple(xy_list[3]),
                               tuple(xy_list[0])
                               ],
                              width=2, fill="green")
                    draw.line([tuple(shrink_xy_list[0]),
                               tuple(shrink_xy_list[1]),
                               tuple(shrink_xy_list[2]),
                               tuple(shrink_xy_list[3]),
                               tuple(shrink_xy_list[0])
                               ],
                              width=2, fill="blue")
                vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
                      [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
                for q_th in range(2):
                    draw.line([tuple(xy_list[vs[long_edge][q_th][0]]),
                               tuple(shrink_1[vs[long_edge][q_th][1]]),
                               tuple(shrink_1[vs[long_edge][q_th][2]]),
                               tuple(xy_list[vs[long_edge][q_th][3]]),
                               tuple(xy_list[vs[long_edge][q_th][4]])],
                              width=3, fill='yellow')
            if cfg.gen_origin_img:
                im.save(os.path.join(train_image_dir, o_img_fname))
            np.save(os.path.join(
                train_label_dir,
                o_img_name + '.npy'),
                xy_list_array)
            if draw_gt_quad:
                show_gt_im.save(os.path.join(show_gt_image_dir, o_img_fname))
            train_val_set.append('{},{},{}\n'.format(o_img_fname,
                                                     d_width,
                                                     d_height))

    train_img_list = os.listdir(train_image_dir)
    print('\nfound %d train images.' % len(train_img_list))
    train_label_list = os.listdir(train_label_dir)
    print('found %d train labels.' % len(train_label_list))

    random.shuffle(train_val_set)
    val_count = int(cfg.validation_split_ratio * len(train_val_set))
    with open(os.path.join(data_dir, cfg.val_fname), 'w') as f_val:
        f_val.writelines(train_val_set[:val_count])
    with open(os.path.join(data_dir, cfg.train_fname), 'w') as f_train:
        f_train.writelines(train_val_set[val_count:])




if __name__ == '__main__':
    preprocess()
