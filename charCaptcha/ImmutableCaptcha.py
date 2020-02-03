# -*- coding=utf-8 -*-
import os
from keras.preprocessing import image
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import tensorflow
'''
验证码类型（高内存占用，训练速度较快版本）：
    英文数字型验证码 识别字符长度不可变
答主资源：
    图片：6000张
    成功率：100%
    内存：1.8G
    显卡：1080ti
    显存：6G
    训练速度：4秒一轮
资源不足的童鞋，后面会出一个小资源版本
'''

# 以下是配置信息，耐心看
'''
验证码可能存在的字符
备注：如果大小写区分请填写 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
'''
characters = '0123456789abcdefghijklmnopqrstuvwxyz'
print('==识别含有字符==')
print(characters)

'''
验证码所在地址
绝对路径相对路径均可
图片必须 正确验证码_随机字符串 命名
示例：2a3mx_00cf9954da274f1eb91407894902482e.jpg
'''
print('==验证码路径==')
datapath = 'E:/captcha/alipay'
print(datapath)

'''
验证码长度
默认5
'''
print('==验证码长度==')
n_len = 4
print(n_len)

'''
模型保存路径
'''
savepath = 'alipay.h5'
'''
以上是必须修改的内容
下面可根据自己情况修改
'''

'''
内存资源丰富可以按实际图片大小来
图片：宽、高 (高是2的次方)
'''
width = 100
height = 32

'''
图片色彩通道数
'''
colorChannel = 3

'''
训练轮次
成功率不够高可以适当提高轮次
耗时也会增加
'''
epochs = 15

'''
批次数量
越大训练的速度越快、消耗资源越多
如果gpu出现oom显存不足，适当调低这个
'''
batch_size = 256

n_class = len(characters)

#这块可注释
tf_config = tensorflow.ConfigProto()
tf_config.gpu_options.allow_growth = True # 自适应
session = tensorflow.Session(config=tf_config)

# 获取目录下样本列表
image_list = []


for item in os.listdir(datapath):
    image_list.append(item)
np.random.shuffle(image_list)
print('==读取到图片数量==')
print(len(image_list))

# 创建数组，储存图片信息。结构为(x, 高度, 宽度, 3)，x代表样本个数。
X = np.zeros((len(image_list), height, width, colorChannel), dtype=np.uint8)
# 创建数组，储存标签信息
y = [np.zeros((len(image_list), n_class), dtype=np.uint8) for i in range(n_len)]
char_indices = dict((c, i) for i, c in enumerate(characters))


# 验证码字符串转数组
def captcha_to_vec(captcha, y, i):
    # 创建一个长度为 字符个数 * 字符种数 长度的数组
    for j, ch in enumerate(captcha):
        y[j][i, :] = 0
        y[j][i, characters.find(ch)] = 1
    return y


for i, img in enumerate(image_list):
    img_path = datapath + "/" + img
    # 读取图片
    raw_img = image.load_img(img_path, target_size=(height, width))
    # 讲图片转为np数组
    X[i] = raw_img
    # 讲标签转换为数组进行保存
    # print(img)
    y = captcha_to_vec(img.split('_')[0], y, i)



#网络搭建、训练过程、保存模型
input_tensor = Input((height, width, 3))
x = input_tensor
for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
    for j in range(n_cnn):
        x = Conv2D(32 * 2 ** min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = MaxPooling2D(2)(x)

x = Flatten()(x)
x = [Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(n_len)]
model = Model(inputs=input_tensor, outputs=x)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(1e-3, amsgrad=True),
              metrics=['accuracy'])

model.fit(X, y, batch_size=batch_size, epochs=epochs)
model.save(savepath)
'''
使用这个方法保存模型，模型大小会大大减小
读取模型方法有所不同
默认关闭
'''
# model.save_weights(savepath)
