# coding:utf-8
"""
CMD:
    python run.py --gpus 0 --save_model_name mobilenetv2 --data_dir sample_ready --base_model_name mobilenetv2 --mode valid
"""

import keras.backend as K
import tensorflow as tf

import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import l2
from keras.preprocessing.image import *
from keras.utils.vis_utils import model_to_dot, plot_model

import random
import os
import cv2
import json
from tqdm import tqdm
from glob import glob
import pathos.multiprocessing as multiprocessing

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from collections import Counter
from keras import backend as K
from keras.utils import multi_gpu_model

from keras_efficientnets import efficientnet
from keras_efficientnets import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3

import argparse

import utils

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0,1,2,3,4')
    parser.add_argument('--save_model_name', type=str, required=True, 
                        help='The model name is used to save h5 model')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='The filename path is relative to the base data dir')
    parser.add_argument('--test_data_dir', type=str, default='test_data',
                        help='The test data path is relative to the base data dir')
    parser.add_argument('--base_model_name', type=str, required=True,
                        choices=['nasnetlarge', 'mobilenetv2', 'xception', 'efficientnet'],
                        help='The pretrained model name')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['train', 'valid', 'test'],
                        help='The script run mode to do train or valid or test')
    return parser.parse_args()


def f(index):
    return index, cv2.resize(cv2.imread(os.path.join(base_data_dir, fnames[index])), (args.width, args.width))


class Generator():
    def __init__(self, X, y, batch_size=32, aug=False):
        def generator():
            idg = ImageDataGenerator(horizontal_flip=True,
                                     rotation_range=20,
                                     zoom_range=0.2)
            while True:
                for i in range(0, len(X), batch_size):
                    X_batch = X[i:i+batch_size].copy()
                    y_barch = [x[i:i+batch_size] for x in y]
                    if aug:
                        for j in range(len(X_batch)):
                            X_batch[j] = idg.random_transform(X_batch[j])
                    yield X_batch, y_barch
        self.generator = generator()
        self.steps = len(X) // batch_size + 1


def acc(y_true, y_pred):
    index = tf.reduce_any(y_true > 0.5, axis=-1)
    res = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    index = tf.cast(index, tf.float32)
    res = tf.cast(res, tf.float32)
    return tf.reduce_sum(res * index) / (tf.reduce_sum(index) + 1e-7)


def create_model(args, label_count):
    if args.base_model_name == 'nasnetlarge':
        base_model = NASNetLarge(weights='imagenet', input_shape=(args.width,args.width,3), include_top=False, pooling='avg')
        preprocess_func = nasnet.preprocess_input
    elif args.base_model_name == 'mobilenetv2':
        base_model = MobileNetV2(weights='imagenet', input_shape=(args.width,args.width,3), include_top=False, pooling='avg')
        preprocess_func = mobilenet_v2.preprocess_input
    elif args.base_model_name == 'xception':
        base_model = Xception(weights='imagenet', input_shape=(args.width,args.width,3), include_top=False, pooling='avg')
        preprocess_func = xception.preprocess_input
    elif args.base_model_name == 'inceptionv3':
        base_model = InceptionV3(weights='imagenet', input_shape=(args.width,args.width,3), include_top=False, pooling='avg')
        preprocess_func = inception_v3.preprocess_input
    elif args.base_model_name == 'efficientnet':
        base_model = EfficientNetB3(weights='imagenet', input_shape=(args.width,args.width,3), include_top=False)
        preprocess_func = efficientnet.preprocess_input
    else:
        raise Exception('Unsupported base model: %s' % base_model_name)

    input_tensor = Input((args.width, args.width, 3))
    print("input_tensor", input_tensor)
    x = input_tensor
    x = Lambda(preprocess_func)(x)
    x = base_model(x)
    if args.base_model_name == 'efficientnet':
        x = GlobalAveragePooling2D(data_format='channels_last')(x)
    x = Dropout(0.5)(x)
    x = [Dense(count, activation='softmax', name=name)(x) for name, count in label_count.items()]
    model = Model(input_tensor, x)
    return model


def pprint_confusion_matrix(labels, data):
    row_format ="{:>20}" * (len(labels) + 1)
    print(row_format.format("", *labels))
    for gt, row in zip(labels, data):
        print(row_format.format(gt, *row))


if __name__ == '__main__':
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

    if args.base_model_name == 'nasnetlarge':
        args.width = 331
    elif args.base_model_name == 'mobilenetv2':
        args.width = 224
    elif args.base_model_name in ['xception', 'inceptionv3']:
        args.width = 299
    elif args.base_model_name == 'efficientnet':
        args.width = 224 #300
    else:
        raise Exception('Unsupported model: %s' % args.base_model_nam_name)

    # 读取数据集
    if args.mode == 'train':
        df = pd.read_csv(os.path.join(args.data_dir, 'Annotations/label.csv'), header=None)
        df.columns = ['filename', 'label_name', 'label']
        df = df.sample(frac=1).reset_index(drop=True) # shuffle

        df.label_name = df.label_name.str.replace('_labels', '')

        print(df.head())
        counts = Counter(df.label_name)
        print(counts)
        
        sorted_key_by_alpha = sorted(counts.keys())
        label_count = dict([(x, len(df[df.label_name == x].label.values[0])) for x in sorted_key_by_alpha])#dict([(x, len(df[df.label_name == x].label.values[0])) for x in counts.keys()])
        label_names = list(label_count.keys())
        print("label_count:", label_count)

        fnames = df['filename'].values

        # 生成y
        # y是一个list, 分为几个part, 表示n个case在每个part的label
        # 具体到一个part时，有可能全0，说明该case没有这部分part；有可能存在一个1，说明该case在这部分有一个label
        n = len(df)
        y = [np.zeros((n, label_count[x])) for x in label_count.keys()]
        for i in range(n):
            label_name = df.label_name[i]
            label = df.label[i]
            # one case can have multi-part label, but the case should appear multi times in label_csv, so actually the multi times are viewed as different cases.
            y[label_names.index(label_name)][i, label.find('y')] = 1 # one part only have one "1" label

        # 读取图片
        X = np.zeros((n, args.width, args.width, 3), dtype=np.uint8)
        base_data_dir = args.data_dir
        with multiprocessing.Pool(8) as pool:
            with tqdm(pool.imap_unordered(f, range(n)), total=n) as pbar:
                for i, img in pbar:
                    X[i] = img[:,:,::-1]

        # 训练集和验证集
        n_train = int(n*0.8)

        X_train = X[:n_train]
        X_valid = X[n_train:]
        y_train = [x[:n_train] for x in y]
        y_valid = [x[n_train:] for x in y]

        gen_train = Generator(X_train, y_train, batch_size=24, aug=True)
    else:
        label_json_file = 'category.json'#'tag-coats-jackets.json'
        data = json.load(open(label_json_file))
        label_mapping = {}
        for k, v in data.items():
            general_label_name = k #label_json_file.split('/')[-1][:-5]
            tmp = sorted(list(v.items()), key=lambda i: i[1].index('y'))
            tmp = [i for i,j in tmp]
            label_mapping[general_label_name] = tmp
        print("label mapping:", label_mapping)
        #label_mapping = {'upper_body': ['POLO衫', '休闲衬衣', '西装衬衣', '长袖T恤', '短袖T恤', '卫衣', '运动上装'],
        #                 'lower_body': ['西装裤', '短裤']
        #                }
        label_count = {}
        sorted_key_by_alpha = sorted(label_mapping.keys())
        for k in sorted_key_by_alpha:
            label_count[k] = len(label_mapping[k])
        print("label_count:", label_count)
        label_names = list(label_count.keys())

    # 搭建模型
    model = create_model(args, label_count)

    if args.mode != 'train':
        print('model_%s.h5' % args.save_model_name)
        model.load_weights('model_%s.h5' % args.save_model_name)
        #model = load_model('model_%s.h5' % args.save_model_name)
    else:
        plot_model(model, show_shapes=True, to_file='model_%s.png' % args.save_model_name)
        #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

        if n_gpus > 1:
            print("use multi-gpu")
            model2 = multi_gpu_model(model, n_gpus)
            training_model = model2
        else:
            training_model = model


    # 训练过程
    def training():
        print("frist training step: lr=1e-4")
        training_model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=[acc])
        training_model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=2, validation_data=(X_valid, y_valid))
        print("first training finish.")

        print("second training step: lr=1e-5")
        training_model.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=[acc])
        training_model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=3, validation_data=(X_valid, y_valid))
        print("second training finish.")

        print("third training step: lr=1e-6")
        training_model.compile(optimizer=Adam(1e-6), loss='categorical_crossentropy', metrics=[acc])
        training_model.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=1, validation_data=(X_valid, y_valid))
        print("third training finish.")
    
    if args.mode == 'train':
        training()
        # 保存模型
        save_model_path = 'model_%s.h5' % args.save_model_name
        print("save model as %s" % save_model_path)
        model.save_weights(save_model_path)

    # 验证过程
    def valid(X_valid, y_valid, general_category_num, save_fig=False):
        #counts = Counter(df.label_name)
        #general_category_num = len(label_count)
        y_pred = model.predict(X_valid, batch_size=128, verbose=1)
        a = np.array([x.any(axis=-1) for x in y_valid]).T.astype('uint8')
        b = [np.where(a[:,x]==1)[0] for x in range(general_category_num)]
        np.set_printoptions(precision=2)
        if not isinstance(y_pred, list):
            y_pred = [y_pred]
        for c in range(general_category_num):
            y_pred2 = y_pred[c][b[c]].argmax(axis=-1)
            y_true2 = y_valid[c][b[c]].argmax(axis=-1)
            print(label_names[c], (y_pred2 == y_true2).mean())
            if save_fig:
                utils.plot_confusion_matrix(y_true2, y_pred2, 
                    classes=label_mapping[label_names[c]], normalize=True, 
                    title='%s_confusion_matrix_%s' % (args.save_model_name, label_names[c]))
        #s = 0
        #n = 0
        #for c in range(general_category_num):
        #    y_pred2 = y_pred[c][b[c]].argmax(axis=-1)
        #    y_true2 = y_valid[c][b[c]].argmax(axis=-1)
        #    s += counts[label_names[c]] * (y_pred2 == y_true2).mean()
        #    n += counts[label_names[c]]
        #print("overall:", s / n)
    
    # 测试过程
    def test():
        y_pred = model.predict(X_test, batch_size=128, verbose=1)
        if not isinstance(y_pred, list):
            y_pred = [y_pred]
        outputs = []
        for i in range(n_test):
            problem_name = df_test.label_name[i].replace('_labels', '')
            problem_index = label_names.index(problem_name)
            probs = y_pred[problem_index][i]
            print("probs", probs)
            outputs.append(label_mapping[problem_name][np.argmax(probs)])
            df_test.label[i] = ';'.join(np.char.mod('%.8f', probs))
        df_test['output'] = outputs
        fname_csv = 'pred_test.csv'
        df_test.to_csv(fname_csv, index=None, header=None)

    if args.mode == 'train':
        print("validation after training start")
        valid(X_valid, y_valid, len(label_count))
    else:
        # 加载测试数据
        base_data_dir = args.test_data_dir
        df_test = pd.read_csv(os.path.join(base_data_dir, 'Annotations/label.csv'), header=None)
        df_test.columns = ['filename', 'label_name', 'label']
        df_test.label_name = df_test.label_name.str.replace('_labels', '')
        fnames = df_test.filename
        n_test = len(df_test)
        print(df_test.head())
        X_test = np.zeros((n_test, args.width, args.width, 3), dtype=np.uint8)
        # if we use f(), please assign base_data_dir and fnames first
        with multiprocessing.Pool(12) as pool:
            with tqdm(pool.imap_unordered(f, range(n_test)), total=n_test) as pbar:
                for i, img in pbar:
                    X_test[i] = img[:,:,::-1]
        # 生成y_test
        n = len(df_test)
        y_test = [np.zeros((n, label_count[x])) for x in label_count.keys()]
        for i in range(n):
            label_name = df_test.label_name[i]
            label = df_test.label[i]
            # one case can have multi-part label, but the case should appear multi times in label_csv, so actually the multi times are viewed as different cases.
            y_test[label_names.index(label_name)][i, label.find('y')] = 1 # one part only have one "1" label
        if args.mode == 'valid':
            print("independently validation start")
            valid(X_test, y_test, len(label_count), save_fig=True)
        else:
            print("testing start")
            # assign base_data_dir and fnames before multiprocessing
            test()
    print("finish")
