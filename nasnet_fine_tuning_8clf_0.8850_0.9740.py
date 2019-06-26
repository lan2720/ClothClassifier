#!/usr/bin/env python
# coding: utf-8

# In[1]:
import sys

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
model_name = 'nasnet_fine_tuning_8clf_2'

import keras.backend as K
import tensorflow as tf

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


# # 导入必要的库

# In[2]:


import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.applications import *
from keras.regularizers import l2

from keras.preprocessing.image import *

import matplotlib.pyplot as plt
import random
import os
import cv2
from tqdm import tqdm
from glob import glob
import multiprocessing

from sklearn.model_selection import train_test_split
from collections import Counter
from keras import backend as K
from keras.utils import multi_gpu_model

#from IPython.display import display, Image

#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# # 读取数据集

# In[3]:


#df = pd.read_csv('fashionAI_attributes_train1/Annotations/label.csv', header=None)
df = pd.read_csv('sample_ready/Annotations/label.csv', header=None)
df.columns = ['filename', 'label_name', 'label']
df = df.sample(frac=1).reset_index(drop=True) # shuffle

df.label_name = df.label_name.str.replace('_labels', '')

#display(df.head())
c = Counter(df.label_name)
#c


# In[4]:


label_count = dict([(x, len(df[df.label_name == x].label.values[0])) for x in c.keys()])
label_names = list(label_count.keys())
#display(label_count)


# ## 生成 y

# In[10]:


fnames = df['filename'].values
width = 331
n = len(df)
y = [np.zeros((n, label_count[x])) for x in label_count.keys()]
for i in range(n):
    label_name = df.label_name[i]
    label = df.label[i]
    y[label_names.index(label_name)][i, label.find('y')] = 1


# ## 读取图片

# In[11]:


def f(index):
    #return index, cv2.resize(cv2.imread('fashionAI_attributes_train1/'+fnames[index]), (width, width))
    return index, cv2.resize(cv2.imread('sample_ready/'+fnames[index]), (width, width))

X = np.zeros((n, width, width, 3), dtype=np.uint8)
with multiprocessing.Pool(8) as pool:
    with tqdm(pool.imap_unordered(f, range(n)), total=n) as pbar:
        for i, img in pbar:
            X[i] = img[:,:,::-1]

print("ok")
#sys.exit()
# In[7]:


n_train = int(n*0.8)

X_train = X[:n_train]
X_valid = X[n_train:]
y_train = [x[:n_train] for x in y]
y_valid = [x[n_train:] for x in y]


# In[8]:


def display_images(imgs, w=8, h=4, figsize=(24, 12)):
    plt.figure(figsize=figsize)
    for i in range(w*h):
        plt.subplot(h, w, i+1)
        plt.imshow(imgs[i])


# In[9]:


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


# In[10]:


gen_train = Generator(X_train, y_train, batch_size=32, aug=True)

print("train data generator ready")


def acc(y_true, y_pred):
    index = tf.reduce_any(y_true > 0.5, axis=-1)
    res = tf.equal(tf.argmax(y_true, axis=-1), tf.argmax(y_pred, axis=-1))
    index = tf.cast(index, tf.float32)
    res = tf.cast(res, tf.float32)
    return tf.reduce_sum(res * index) / (tf.reduce_sum(index) + 1e-7)



base_model = NASNetLarge(weights='imagenet', input_shape=(width, width, 3), include_top=False, pooling='avg')

input_tensor = Input((width, width, 3))
x = input_tensor
x = Lambda(nasnet.preprocess_input)(x)
x = base_model(x)
x = Dropout(0.5)(x)
x = [Dense(count, activation='softmax', name=name)(x) for name, count in label_count.items()]

model = Model(input_tensor, x)
print("模型搭建完毕")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
print("模型编译完毕")
#from IPython.display import SVG
#from keras.utils.vis_utils import model_to_dot, plot_model

#plot_model(model, show_shapes=True, to_file='model_simple.png')
#SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


# In[15]:


model2 = multi_gpu_model(model, n_gpus)
print("多GPU模型搭建完毕")

# In[16]:


# opt = SGD(1e-3, momentum=0.9, nesterov=True, decay=1e-5)


# In[17]:


model2.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=[acc])
print("多GPU模型第一次编译 lr=1e-4")
model2.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=2, validation_data=(X_valid, y_valid))
print("多GPU模型第一个fit")


# In[18]:

model2.compile(optimizer=Adam(1e-5), loss='categorical_crossentropy', metrics=[acc])
print("多GPU模型第二次编译 lr=1e-5")
model2.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=3, validation_data=(X_valid, y_valid))
print("多GPU模型第二次fit")

# In[19]:


model2.compile(optimizer=Adam(1e-6), loss='categorical_crossentropy', metrics=[acc])
print("多GPU模型第三次编译 lr=1e-6")
model2.fit_generator(gen_train.generator, steps_per_epoch=gen_train.steps, epochs=1, validation_data=(X_valid, y_valid))
print("多GPU模型第三次fit")


# # 计算验证集准确率

# In[20]:


model.save('model_%s.h5' % model_name)
print("保存模型完毕")

# In[21]:

print("验证集测试")
y_pred = model2.predict(X_valid, batch_size=128, verbose=1)
a = np.array([x.any(axis=-1) for x in y_valid]).T.astype('uint8')
b = [np.where((a == np.eye(8)[x]).all(axis=-1))[0] for x in range(8)]
for c in range(8):
    y_pred2 = y_pred[c][b[c]].argmax(axis=-1)
    y_true2 = y_valid[c][b[c]].argmax(axis=-1)
    print(label_names[c], (y_pred2 == y_true2).mean())


# In[ ]:





# In[22]:


counts = Counter(df.label_name)
counts


# In[23]:


s = 0
n = 0
for c in range(8):
    y_pred2 = y_pred[c][b[c]].argmax(axis=-1)
    y_true2 = y_valid[c][b[c]].argmax(axis=-1)
    s += counts[label_names[c]] * (y_pred2 == y_true2).mean()
    n += counts[label_names[c]]
print(s / n)


# # 在测试集上预测

df_test = pd.read_csv('rank/Tests/question.csv', header=None)
df_test.columns = ['filename', 'label_name', 'label']

fnames_test = df_test.filename

n_test = len(df_test)
df_test.head()


def f(index):
    return index, cv2.resize(cv2.imread('rank/'+fnames_test[index]), (width, width))

X_test = np.zeros((n_test, width, width, 3), dtype=np.uint8)
with multiprocessing.Pool(12) as pool:
    with tqdm(pool.imap_unordered(f, range(n_test)), total=n_test) as pbar:
        for i, img in pbar:
            X_test[i] = img[:,:,::-1]


y_pred = model2.predict(X_test, batch_size=128, verbose=1)

for i in range(n_test):
    problem_name = df_test.label_name[i].replace('_labels', '')
    problem_index = label_names.index(problem_name)
    probs = y_pred[problem_index][i]
    df_test.label[i] = ';'.join(np.char.mod('%.8f', probs))

fname_csv = 'pred_%s.csv' % model_name
df_test.to_csv(fname_csv, index=None, header=None)
