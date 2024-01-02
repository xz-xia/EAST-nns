import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler,StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras import initializers
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten
import numpy as np
from tensorflow.keras.models import load_model
import os
import math
from scipy import interpolate
#%%
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
#%%
R=np.linspace(1.2,2.8,129)
Z=np.linspace(-1.4,1.4,129)
dr = R[1]-R[0]
dz = Z[1]-Z[0]
#%%
x_train = np.load('D:/科大研究生生活/文章/2023.10后补充工作/数据/按炮划分数据/第三次筛选/train_input.npy')
y_train = np.load('D:/科大研究生生活/文章/2023.10后补充工作/数据/按炮划分数据/第三次筛选/train_output.npy')
x_val = np.load('D:/科大研究生生活/文章/2023.10后补充工作/数据/按炮划分数据/第三次筛选/val_input.npy')
y_val = np.load('D:/科大研究生生活/文章/2023.10后补充工作/数据/按炮划分数据/第三次筛选/val_output.npy')
x_test = np.load('D:/科大研究生生活/文章/2023.10后补充工作/数据/按炮划分数据/第三次筛选/test_input.npy')
y_test = np.load('D:/科大研究生生活/文章/2023.10后补充工作/数据/按炮划分数据/第三次筛选/test_output.npy')
#%%
x = np.vstack((x_train,x_val))
y = np.vstack((y_train,y_val))
#%%
train_sample_weight = np.load('D:/科大研究生生活/文章/2023.10后补充工作/数据/按炮划分数据/第三次筛选/train_sample_weight.npy')
val_sample_weight = np.load('D:/科大研究生生活/文章/2023.10后补充工作/数据/按炮划分数据/第三次筛选/val_sample_weight.npy')
test_sample_weight = np.load('D:/科大研究生生活/文章/2023.10后补充工作/数据/按炮划分数据/第三次筛选/test_sample_weight.npy')
#%%
MaxAbs_Scaler = MaxAbsScaler()
x_train = MaxAbs_Scaler.fit_transform(x_train)
x_test = MaxAbs_Scaler.transform(x_test)
x_val = MaxAbs_Scaler.transform(x_val)
#%%
y_train = np.array(y_train).reshape(len(x_train),129*129)
y_val = np.array(y_val).reshape(len(x_val),129*129)
#%%
#初始化参数
in_features = 75
out_features = 129*129
#初始化参数
epoch = 100
batch_size = 64
initializer = initializers.TruncatedNormal(mean=0, stddev=0.05, seed = 0)
#%%
#网络模型
model = Sequential()
#输入层input_shape = (in_features,)
model.add(Dense(60,input_shape = (in_features,) ,kernel_initializer=initializer,kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-4)))
#kernel_constraint= keras.constraints.MinMaxNorm()
#tf.keras.layers.LayerNormalization()
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('tanh'))
model.add(Dropout(0.5))
#隐藏层1
model.add(Dense(120,kernel_initializer=initializer,kernel_regularizer=keras.regularizers.l2(1e-4)))
#tf.keras.layers.LayerNormalization()
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('tanh'))
model.add(Dropout(0.5))
#隐藏层2
model.add(Dense(120,kernel_initializer=initializer,kernel_regularizer=keras.regularizers.l2(1e-4)))
#tf.keras.layers.LayerNormalization()
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('tanh'))
model.add(Dropout(0.5))
#隐藏层3
model.add(Dense(240,kernel_initializer=initializer,kernel_regularizer=keras.regularizers.l2(1e-4)))
#tf.keras.layers.LayerNormalization()
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('tanh'))
model.add(Dropout(0.5))
#输出层
model.add(Dense(out_features,kernel_initializer=initializer,kernel_regularizer=keras.regularizers.l2(1e-4)))
#%%
lr =0.0001
#%%
weight = np.zeros((1,129,129))
weight[:,36:98, 14:100]=1
weight = weight.reshape(-1,129*129)
tensor_weight = tf.constant(weight,dtype=np.float32)
#%%
def myloss(y_true,y_pred):
    MSE_loss_new = tf.reduce_mean(tf.square(tf.multiply(y_pred,tensor_weight)-tf.multiply(y_true,tensor_weight)),axis=-1)
    MSE_loss = tf.reduce_mean(tf.square(y_true-y_pred))
    total_loss = MSE_loss + MSE_loss_new
    return total_loss
#%%
#损失函数和优化器
opt = tf.optimizers.Adam(learning_rate = lr,clipvalue=0.5,)
model.compile(optimizer=opt , loss=myloss,run_eagerly=True)
#%%
train_sample_weight = train_sample_weight.reshape(-1,1)
val_sample_weight = val_sample_weight.reshape(-1,1)
test_sample_weight = test_sample_weight.reshape(-1,1)
#%%
#训练网络
print("开始训练。。。")
M = model.fit(x_train,y_train,sample_weight=train_sample_weight,validation_data=(x_val,y_val,val_sample_weight),epochs=epoch,batch_size=batch_size)

#%%
from tensorflow.keras.models import load_model
#%%
model.save('D:/科大研究生生活/文章/2023.10后补充工作/模型/第三次筛选/conv1d_relu.h5')