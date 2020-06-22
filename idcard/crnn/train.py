import os
import random
from keras import callbacks
from keras import initializers
from keras import backend as K
from keras.models import Model
from keras.layers import (Input, BatchNormalization, Activation, Conv2D, MaxPooling2D,
                          Permute, Dense, LSTM, Lambda, TimeDistributed, Flatten, Bidirectional)
import numpy as np
import cv2
from tqdm import tqdm
from aug import DataAugmentation

def ctc_loss_layer(args):
    y_true, y_pred, pred_length, label_length = args
    batch_cost = K.ctc_batch_cost(y_true, y_pred, pred_length, label_length) #损失函数 最大似然估计
    return batch_cost

def fake_ctc_loss(y_true, y_pred):
    return y_pred

class CBC:

    @staticmethod #静态方法
    def build(img_size, num_classes, max_label_length, is_training=True):

        initializer = initializers.he_normal()
        img_width, img_height = img_size

        def PatternUnits(inputs, index, activation="relu"): #规范化和激活
            inputs = BatchNormalization(name="BN_%d" % index)(inputs) #BN算法对数据处理 该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1 收敛速度更快
            inputs = Activation(activation, name="Relu_%d" % index)(inputs) #激活函数 输入信号转换成输出信号

            return inputs
        #cnn
        inputs = Input(shape=(img_height, img_width, 1), name='img_inputs') #32*256*1
        x = Conv2D(64, (3, 3), padding="same", kernel_initializer=initializer, name='Conv2d_1')(inputs) #32*256*64 卷积层1 过滤器数量、宽高、输入输出形状相同、卷积核初始化、层名
        x = PatternUnits(x, 1)
        x = MaxPooling2D(strides=2, name='Maxpool_1')(x) #16*128*64 池化层1 步长、层名
        x = Conv2D(128, (3, 3), padding="same", kernel_initializer=initializer, name='Conv2d_2')(x) #16*128*128
        x = PatternUnits(x, 2)
        x = MaxPooling2D(strides=2, name='Maxpool_2')(x) #8*64*128

        x = Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name='Conv2d_3')(x) #8*64*256
        x = PatternUnits(x, 3)
        x = Conv2D(256, (3, 3), padding="same", kernel_initializer=initializer, name='Conv2d_4')(x) #8*64*256
        x = PatternUnits(x, 4)
        x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='Maxpool_3')(x) #4*64*256 垂直缩小

        x = Conv2D(512, (3, 3), padding="same", kernel_initializer=initializer, name='Conv2d_5')(x)
        x = PatternUnits(x, 5)
        x = Conv2D(512, (3, 3), padding="same", kernel_initializer=initializer, name='Conv2d_6')(x) #4*64*512
        x = PatternUnits(x, 6)
        x = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), name='Maxpool_4')(x) #2*64*512

        x = Conv2D(512, (2, 2), padding='same', activation='relu', kernel_initializer=initializer, name='Conv2d_7')(x)
        x = PatternUnits(x, 7)
        conv_output = MaxPooling2D(pool_size=(2, 1), name="Conv_output")(x) #1*64*512
        x = Permute((2, 3, 1), name='Permute')(conv_output) #64*512*1 置换维度
        #rnn BLSTM
        rnn_input = TimeDistributed(Flatten(), name='Flatten_by_time')(x) #64*512 按时间将输入一维化
        y = Bidirectional(LSTM(256, kernel_initializer=initializer, return_sequences=True), #输出维度、矩阵初始化器、返回整个序列
                          merge_mode='sum', name='LSTM_1')(rnn_input) #双向封装器 前向后向输出结合方式
        y = BatchNormalization(name='BN_8')(y) #归一化
        y = Bidirectional(LSTM(256, kernel_initializer=initializer, return_sequences=True), name='LSTM_2')(y)

        y_pred = Dense(num_classes, activation='softmax', name='y_pred')(y) #神经元个数、激活函数
        y_true = Input(shape=[max_label_length], name='y_true') #实例化
        y_pred_length = Input(shape=[1], name='y_pred_length')
        y_true_length = Input(shape=[1], name='y_true_length')
        ctc_loss_output = Lambda(ctc_loss_layer, output_shape=(1,), name='ctc_loss_output')(#封装成layer对象
            [y_true, y_pred, y_pred_length, y_true_length])
        base_model = Model(inputs=inputs, outputs=y_pred)#在网络中封装具有训练和推理特征的对象
        base_model.summary()
        model = Model(inputs=[y_true, inputs, y_pred_length, y_true_length], outputs=ctc_loss_output)
        model.summary()
        if is_training:
            return model
        else:
            return base_model

    @staticmethod #静态方法
    def train(model, src_dir, save_dir, img_size, batch_size, max_label_length, aug_nbr, epochs):
        print("[*] Setting up for checkpoints.")
        ckpt = callbacks.ModelCheckpoint(save_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5", #回调保存模型
                                         save_weights_only=True, save_best_only=True)#只保存模型权重、监测值有改进时才会保存当前的模型
        reduce_lr_cbk = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3)#3个epoch val_loss不再提示则减小学习速率
        print("[*] Setting up for compiler.")
        model.compile(optimizer='adam', loss={'ctc_loss_output': fake_ctc_loss})#优化器、损失函数类型
        print("[*] Preparing data generator.")
        train_list, val_list = train_val_split(src_dir)
        train_gen = DataGenerator(train_list,
                                  img_shape=img_size,
                                  down_sample_factor=4,
                                  batch_size=batch_size,
                                  max_label_length=max_label_length,
                                  max_aug_nbr=aug_nbr,
                                  width_shift_range=15,
                                  height_shift_range=10,
                                  zoom_range=12,
                                  shear_range=15,
                                  rotation_range=20,
                                  blur_factor=5,
                                  add_noise_factor=0.01)
        val_gen = DataGenerator(val_list, img_size, batch_size, max_label_length)
        print("[*] Training start!")
        model.fit_generator(generator=train_gen.flow(),
                            steps_per_epoch=200,
                            validation_data=val_gen.flow(),
                            validation_steps=val_gen.data_nbr // batch_size,
                            callbacks=[ckpt, reduce_lr_cbk],
                            epochs=epochs)#使用fit_generator进行训练
        print("[*] Training finished!")
        model.save(save_dir + "crnn_model.h5")
        print("[*] Model has been successfully saved in %s!" % save_dir)
        return 0

class DataGenerator:

    def __init__(self,
                 data_list,
                 img_shape,
                 batch_size,
                 down_sample_factor=4,
                 max_label_length=26,
                 max_aug_nbr=1,
                 width_shift_range=0,
                 height_shift_range=0,
                 zoom_range=0,
                 shear_range=0,
                 rotation_range=0,
                 blur_factor=None,
                 add_noise_factor=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 has_wrapped_dataset=None
                 ):
        #数据路径
        self.data_list = data_list
        self.img_w, self.img_h = img_shape
        self.batch_size = batch_size
        self.max_label_length = max_label_length
        self.pre_pred_label_length = int(self.img_w // down_sample_factor)
        self.data_nbr = len(data_list) * max_aug_nbr
        self.local_dataset_path = None

        #从.npz file读数据
        self.load_data = "D:/idcard/crnn/train.npz"
        #定义缩写
        self.param_dict = {
            'wsr': width_shift_range,
            'hsr': height_shift_range,
            'zor': zoom_range,
            'shr': shear_range,
            'ror': rotation_range,
            'blr': blur_factor,
            'nof': add_noise_factor,
            'hfl': horizontal_flip,
            'vfl': vertical_flip
        }
        #是否增强的信号
        self.max_aug_nbr = max_aug_nbr
        #直接加载数据
        if has_wrapped_dataset is not None:
            print("[*] Using local wrapped dataset.")
            self.load_data = np.load(has_wrapped_dataset)
        else:
            sign = data_wrapper(data_list, img_shape, max_label_length, max_aug_nbr, self.param_dict, name="train")
            if isinstance(sign, str):
                self.load_data = np.load(sign)
                self.local_dataset_path = sign
            else:
                self.data, self.labels, self.labels_length = sign

    def flow(self):
        #向training generator提供输入和输出
        pred_labels_length = np.full((self.batch_size, 1), self.pre_pred_label_length, dtype=np.float64)

        while True:
            #随机选择工作范围
            working_start_index = np.random.randint(self.data_nbr - self.batch_size)
            working_end_index = working_start_index + self.batch_size
            if self.load_data is not None:
                #不能一次性从.npz中读取所有这些数据，因为会消耗大量的内存，减慢训练速度。必要的时候剪辑并读取
                working_data = self.load_data["data"][working_start_index: working_end_index]
                working_labels = self.load_data["labels"][working_start_index: working_end_index]
                working_labels_length = self.load_data["labels_length"][working_start_index: working_end_index]
            else:
                working_data = self.data[working_start_index: working_end_index]
                working_labels = self.labels[working_start_index: working_end_index]
                working_labels_length = self.labels_length[working_start_index: working_end_index]
            inputs = {
                "y_true": working_labels,
                "img_inputs": working_data,
                "y_pred_length": pred_labels_length,
                "y_true_length": working_labels_length
            }
            outputs = {"ctc_loss_output": np.zeros((self.batch_size, 1), dtype=np.float64)}

            yield (inputs, outputs)

def train_val_split(src_dir, val_split_ratio=0.1):
    #准备一个完整的图像和标签列表。
    data_path, labels = [], []
    for file in os.listdir(src_dir):
        data_path.append(src_dir + file)
        name, ext = os.path.splitext(file)
        labels.append(name[:4])

    #随机选择“按索引设置值”
    length = len(data_path)
    rand_index = list(range(length))
    random.shuffle(rand_index)
    val_index = rand_index[0: int(val_split_ratio * length)]

    #收集并返回
    train_set, val_set = [], []
    for i in range(length):
        if i in val_index:
            val_set.append((data_path[i], labels[i]))
        else:
            train_set.append((data_path[i], labels[i]))
    return train_set, val_set

def data_wrapper(src_list, img_shape, max_label_length,
                 max_aug_nbr=1, aug_param_dict=None, name="temp"):
    # Initialize some variables
    n = len(src_list) * max_aug_nbr
    img_w, img_h = img_shape
    is_saved = False
    # Status progress bar.
    p_bar = tqdm(total=len(src_list))
    # Create random indexes.
    rand_index = np.random.permutation(n)

    data = np.zeros((n, img_h, img_w))
    labels = np.zeros((n, max_label_length))
    labels_length = np.zeros((n, 1))

    def valid_img(image):
        # Do some common process to image.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, img_shape)
        image = image.astype(np.float32)
        return image

    def valid_label(label_string):
        # Get rid of the empty placeholder '_'.
        # Even it is the head of label.
        res = []
        for ch in label_string:
            if ch == '_':
                continue
            else:
                res.append(int(ch))
        a = len(res)
        for i in range(max_label_length - a):
            res.append(10)  # represent '_'
        # Return res for labels, length for labels_length
        return res, a

    index = 0
    for path, label in src_list:
        img = cv2.imread(path)
        v_lab, v_len = valid_label(label)
        data[rand_index[index]] = valid_img(img)
        labels[rand_index[index]] = v_lab
        labels_length[rand_index[index]][0] = v_len

        if max_aug_nbr != 1 and aug_param_dict is not None and any(aug_param_dict):
            is_saved = True
            #一旦触发数据增强，它将保存在本地
            aug = DataAugmentation(img, aug_param_dict)
            # max_aug_nbr = original_img(.also 1) + augment_img
            for aug_img in aug.feed(max_aug_nbr-1):
                index += 1
                data[rand_index[index]] = valid_img(aug_img)
                # Different augmentation of images, but same labels and length.
                labels[rand_index[index]] = v_lab
                labels_length[rand_index[index]][0] = v_len
        index += 1
        p_bar.update()
    p_bar.close()
    data.astype(np.float64) / 255.0 * 2 - 1
    data = np.expand_dims(data, axis=-1)
    labels.astype(np.float64)

    if is_saved:
        local_path = "%s.npz" % name
        np.savez(local_path, data=data, labels=labels, labels_length=labels_length)
        print("[*] Data with augmentation has been saved.")
        return local_path
    else:
        return data, labels, labels_length