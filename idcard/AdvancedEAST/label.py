import numpy as np
import os
from PIL import Image, ImageDraw
from tqdm import tqdm
from AdvancedEAST import cfg


def point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
    if (p_min[0] <= px <= p_max[0]) and (p_min[1] <= py <= p_max[1]):
        xy_list = np.zeros((4, 2))
        xy_list[:3, :] = quad_xy_list[1:4, :] - quad_xy_list[:3, :]
        # xy_list[0] = [x1-x0, y1-y0]  xy_list[1] = [x2-x1, y2-y1]  xy_list[2] = [x3-x2, y3-y2]
        xy_list[3] = quad_xy_list[0, :] - quad_xy_list[3, :]
        yx_list = np.zeros((4, 2))
        yx_list[:, :] = quad_xy_list[:, -1:-3:-1]
        # yx_list[0] = [y0, x0]  yx_list[1] = [y1, x1] yx_list[2] = [y2, x2] yx_list[3] = [y3, x3]
        a = xy_list * ([py, px] - yx_list)
        # a[0] = [x1-x0, y1-y0] * ([py, px] - [y0, x0])
        #    = [x1-x0, y1-y0] * [py-y0, px-x0]
        #    = [(x1-x0)*(py-y0), (y1-y0)*(px-x0)]
        b = a[:, 0] - a[:, 1]
        # b[0] = (x1-x0)*(py-y0) - (y1-y0)*(px-x0)
        if np.amin(b) >= 0 or np.amax(b) <= 0:
            return True
        else:
            return False
    else:
        return False


def point_inside_of_nth_quad(px, py, xy_list, shrink_1, long_edge):
    nth = -1
    vs = [[[0, 0, 3, 3, 0], [1, 1, 2, 2, 1]],
          [[0, 0, 1, 1, 0], [2, 2, 3, 3, 2]]]
    for ith in range(2):
        quad_xy_list = np.concatenate((
            np.reshape(xy_list[vs[long_edge][ith][0]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][1]], (1, 2)),
            np.reshape(shrink_1[vs[long_edge][ith][2]], (1, 2)),
            np.reshape(xy_list[vs[long_edge][ith][3]], (1, 2))), axis=0)
        p_min = np.amin(quad_xy_list, axis=0)
        p_max = np.amax(quad_xy_list, axis=0)
        if point_inside_of_quad(px, py, quad_xy_list, p_min, p_max):
            if nth == -1:
                nth = ith
            else:
                nth = -1
                break
    return nth

def shrink(xy_list, ratio=cfg.shrink_ratio):
    if ratio == 0.0:
        return xy_list, xy_list
    #计算相邻两点的距离
    diff_1to3 = xy_list[:3, :] - xy_list[1:4, :]
    diff_4 = xy_list[3:4, :] - xy_list[0:1, :]
    diff = np.concatenate((diff_1to3, diff_4), axis=0)
    dis = np.sqrt(np.sum(np.square(diff), axis=-1))
    # 确定是长边还是短边
    long_edge = int(np.argmax(np.sum(np.reshape(dis, (2, 2)), axis=0)))
    short_edge = 1 - long_edge
    r = [np.minimum(dis[i], dis[(i + 1) % 4]) for i in range(4)]
    #计算每条边与x轴的夹角
    diff_abs = np.abs(diff)
    diff_abs[:, 0] += cfg.epsilon
    theta = np.arctan(diff_abs[:, 1] / diff_abs[:, 0])
    # 调用shrink_edge，先缩短两条长边
    temp_new_xy_list = np.copy(xy_list)
    shrink_edge(xy_list, temp_new_xy_list, long_edge, r, theta, ratio)
    shrink_edge(xy_list, temp_new_xy_list, long_edge + 2, r, theta, ratio)
    # 再缩进两条短边
    new_xy_list = np.copy(temp_new_xy_list)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge, r, theta, ratio)
    shrink_edge(temp_new_xy_list, new_xy_list, short_edge + 2, r, theta, ratio)
    return temp_new_xy_list, new_xy_list, long_edge


def shrink_edge(xy_list, new_xy_list, edge, r, theta, ratio=cfg.shrink_ratio):
    if ratio == 0.0:
        return
    start_point = edge
    end_point = (edge + 1) % 4
    #在x方向上移动起点 x=x+缩进比*边长*cosθ
    long_start_sign_x = np.sign(xy_list[end_point, 0] - xy_list[start_point, 0])
    new_xy_list[start_point, 0] = \
        xy_list[start_point, 0] + \
        long_start_sign_x * ratio * r[start_point] * np.cos(theta[start_point])
    # 在y方向上移动起点 y=y+缩进比*边长*sinθ
    long_start_sign_y = np.sign(xy_list[end_point, 1] - xy_list[start_point, 1])
    new_xy_list[start_point, 1] = \
        xy_list[start_point, 1] + \
        long_start_sign_y * ratio * r[start_point] * np.sin(theta[start_point])
    #在x方向上移动终点 x=x-缩进比*边长*cosθ
    long_end_sign_x = -1 * long_start_sign_x
    new_xy_list[end_point, 0] = \
        xy_list[end_point, 0] + \
        long_end_sign_x * ratio * r[end_point] * np.cos(theta[start_point])
    # 在y方向上移动终点 y=y-缩进比*边长*sinθ
    long_end_sign_y = -1 * long_start_sign_y
    new_xy_list[end_point, 1] = \
        xy_list[end_point, 1] + \
        long_end_sign_y * ratio * r[end_point] * np.sin(theta[start_point])

def process_label(data_dir=cfg.data_dir):
    with open(os.path.join(data_dir, cfg.val_fname), 'r') as f_val:
        f_list = f_val.readlines()
    with open(os.path.join(data_dir, cfg.train_fname), 'r') as f_train:
        f_list.extend(f_train.readlines())
    for line, _ in zip(f_list, tqdm(range(len(f_list)))):
        line_cols = str(line).strip().split(',')
        img_name_ext, width, height = line_cols[0].strip(), int(line_cols[1].strip()), \
                                      int(line_cols[2].strip())
        img_name, _ = os.path.splitext(img_name_ext)
        gt = np.zeros((height // cfg.pixel_size, width // cfg.pixel_size, 7))
        train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)
        xy_list_array = np.load(os.path.join(train_label_dir,
                                             img_name + '.npy'))
        train_image_dir = os.path.join(data_dir, cfg.train_image_dir_name)
        with Image.open(os.path.join(train_image_dir, img_name_ext)) as im:
            draw = ImageDraw.Draw(im)
            for xy_list in xy_list_array:
                _, shrink_xy_list, _ = shrink(xy_list, cfg.shrink_ratio)
                shrink_1, _, long_edge = shrink(xy_list, cfg.shrink_side_ratio)
                p_min = np.amin(shrink_xy_list, axis=0)
                p_max = np.amax(shrink_xy_list, axis=0)
                # floor of the float
                ji_min = (p_min / cfg.pixel_size - 0.5).astype(int) - 1
                # +1 for ceil of the float and +1 for include the end
                ji_max = (p_max / cfg.pixel_size - 0.5).astype(int) + 3
                i_min = np.maximum(0, ji_min[1])
                i_max = np.minimum(height // cfg.pixel_size, ji_max[1])
                j_min = np.maximum(0, ji_min[0])
                j_max = np.minimum(width // cfg.pixel_size, ji_max[0])
                for i in range(i_min, i_max):
                    for j in range(j_min, j_max):
                        px = (j + 0.5) * cfg.pixel_size
                        py = (i + 0.5) * cfg.pixel_size
                        if point_inside_of_quad(px, py, shrink_xy_list, p_min, p_max):
                            gt[i, j, 0] = 1
                            line_width, line_color = 1, 'red'
                            ith = point_inside_of_nth_quad(px, py, xy_list, shrink_1, long_edge)
                            vs = [[[3, 0], [1, 2]], [[0, 1], [2, 3]]]
                            if ith in range(2):
                                gt[i, j, 1] = 1
                                if ith == 0:
                                    line_width, line_color = 2, 'yellow'
                                else:
                                    line_width, line_color = 2, 'green'
                                gt[i, j, 2:3] = ith
                                gt[i, j, 3:5] = xy_list[vs[long_edge][ith][0]] - [px, py]
                                gt[i, j, 5:] = xy_list[vs[long_edge][ith][1]] - [px, py]
                            draw.line([(px - 0.5 * cfg.pixel_size,
                                        py - 0.5 * cfg.pixel_size),
                                       (px + 0.5 * cfg.pixel_size,
                                        py - 0.5 * cfg.pixel_size),
                                       (px + 0.5 * cfg.pixel_size,
                                        py + 0.5 * cfg.pixel_size),
                                       (px - 0.5 * cfg.pixel_size,
                                        py + 0.5 * cfg.pixel_size),
                                       (px - 0.5 * cfg.pixel_size,
                                        py - 0.5 * cfg.pixel_size)],
                                      width=line_width, fill=line_color)
            act_image_dir = os.path.join(cfg.data_dir,
                                         cfg.show_act_image_dir_name)
            if cfg.draw_act_quad:
                im.save(os.path.join(act_image_dir, img_name_ext))
        train_label_dir = os.path.join(data_dir, cfg.train_label_dir_name)
        np.save(os.path.join(train_label_dir, img_name + '_gt.npy'), gt)

if __name__ == '__main__':
    process_label()




