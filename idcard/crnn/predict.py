import cv2
import numpy as np
from keras import backend as K
from crnn.train import CBC

num2char = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '_'}

def recognition(img_array, img_shape, model_path,
                       num_classes=11,
                       max_label_length=26,
                       downsample_factor=4): #下采样因子
    img_w, img_h = img_shape
    # 图像处理
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY) #转化成灰度图
    img_array = cv2.resize(img_array, img_shape) #缩放
    img_array = np.expand_dims(img_array, axis=-1) #在数组最后添加数据
    img_array = img_array / 255.0 * 2.0 - 1.0

    img_batch = np.zeros((1, img_h, img_w, 1)) #返回一个4维数组
    img_batch[0, :, :, :] = img_array

    # 预测模型
    model = CBC.build(img_shape, num_classes, max_label_length, is_training=False)
    model.load_weights(model_path) #读取模型
    y_pred = model.predict(img_batch) #批量预测
    y_pred_tensor_list, _ = K.ctc_decode(y_pred, [img_w // downsample_factor]) #解码
    y_pred_tensor = y_pred_tensor_list[0]
    y_pred_labels = K.get_value(y_pred_tensor)
    y_pred_text = ""
    for num in y_pred_labels[0]:
        y_pred_text += num2char[num]

    return y_pred_text
