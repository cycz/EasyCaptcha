# 使用深度学习来破解 captcha 验证码

本项目会通过 Keras 搭建一个深度卷积神经网络来识别 captcha 验证码，建议使用显卡来运行该项目。

本项目完全依赖配置即可训练出可用，高精确率模型，无需关心代码

完全笨蛋式配置，十分钟即可完成验证码训练
   

# 环境
#### python版本

- [x] 3.6





# 项目目录

```
|-- charCaptcha //字符型验证码
|   |-- ImmutableCaptcha.h5 //h5结尾一般是模型文件
|   |-- ImmutableCaptcha.py //固定位数字符型验证码训练入口
|   |-- loadModel
|   |   `-- ImmutableCaptcha_loadModel.py //固定位数字符型验证码读取模型样例
|   `-- pic
|       |-- 2a3mx_00cf9954da274f1eb91407894902482e.jpg
|       `-- jianshe.zip //验证码标记训练集
|-- README.md
`-- requirements.txt
```

#使用方法



```
1、安装python3.6
2、pip install  -r requirements.txt （使用镜像源会更快）
3、选择训练验证码模型入口
4、根据需求修改代码中的配置、运行
5、使用loadModel中的调用模型样例，加载模型
```

#注意事项
1、使用gpu 将requirements.txt中的 tensorflow 修改为tensorflow-gpu （tensorflow安装问题可以自行百度一下）

2、更多验证码类型持续更新


给大家练习的训练集

链接：https://pan.baidu.com/s/1Fxiv5j6eXLcgDOR5G97_lQ 
提取码：44c8