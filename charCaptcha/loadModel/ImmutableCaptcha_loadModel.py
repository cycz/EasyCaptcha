import numpy as np
from keras.preprocessing import image
from keras.models import load_model

'''
所有配置需要和训练时一样
# 控制台输入图片路径
# 参考示例：../pic/2a3mx_00cf9954da274f1eb91407894902482e.jpg
'''

'''
验证码可能存在的字符
备注：如果大小写区分请填写 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
'''
characters = '0123456789abcdefghijklmnopqrstuvwxyz'
print('==识别含有字符==')
print(characters)

'''
模型路径
'''
modelpath = '../alipay.h5'

'''
内存资源丰富可以按实际图片大小来
图片：宽、高
'''
width = 128
height = 32

'''
图片色彩通道数
默认3 如果图片进行灰度处理可以设置为1
'''
colorChannel = 3

X = np.zeros((1, height, width, colorChannel), dtype=np.uint8)
char_indices = dict((c, i) for i, c in enumerate(characters))

def img2x(path):
    raw_img = image.load_img(path, target_size=(height, width))
    X[0] = raw_img
    return X

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:, 0]
    return ''.join([characters[x] for x in y])

#读取模型主要代码
#小内存模型不可用这个方法读取
model = load_model(modelpath)

# 控制台输入图片路径
# 参考示例：../pic/2a3mx_00cf9954da274f1eb91407894902482e.jpg
while (1):
    print('Please input your path:')
    picPath = input()
    if picPath == '' or picPath.isspace():
        print('See you next time!')
        break
    else:
        result = ''
        XXX = img2x(picPath)
        XXX = np.array(XXX)
        for i in model.predict(XXX):
            result += characters[np.argmax(i)]
        print(result)
