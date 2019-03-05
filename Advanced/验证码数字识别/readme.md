# 验证码数字识别
[数据集等](https://github.com/zyq-361/KaggleMachCode/tree/master/match3)


本次竞赛的目的是对由3位数字构成的图像进行识别，例如，当图像中的数字为001时，识别结果为1；当图像中的数字为011时，识别结果为11；当图像中的数字为111时，识别结果为111。



## 赛制介绍

本次比赛系机器学习方向2016级内部竞赛的第三次竞赛（决赛），旨在通过对图像数据进行建模分析，完整地完成比赛过程。通过本次比赛，学生应该掌握对图像数据的基本处理和熟悉计算机视觉领域的基本任务及解决方法等。

### 赛程安排

**时间**：10月31日-11月27日。

**流程**：

1.每位同学注册一个kaggle账号，并登入，然后打开指定链接，参与比赛，最后由小组组长将组员结合为一个team，每位同学可以查看相应的竞赛要求和数据集。

2.从该平台下载数据集，本地调试算法，并在该平台上提交结果，提交完成后，即得相应的分数并可以查看公共排名结果。（注意：公共排名仅供参考，比赛结束后，最终排名以私有排名为准）

3.每支参赛队伍每天有且仅有二次提交机会，高分成绩直接覆盖低分成绩。

4.比赛提供标准训练集（含标签），仅供参赛选手训练算法模型；提供标准测试数据集（无标签），供参赛选手提交评测结果参与排名。比赛中严禁使用外部数据集，一经发现，成绩作废。

5.比赛中需要使用代码管理工具对代码进行管理，并在提交结果之后存储当前代码作为提交成绩依据的记录，比赛结束公布最终排名时需提交取得最高成绩的代码。

6.比赛截止时，将对所有参赛队伍代码进行审核，识别并剔除只靠人工标注而没有算法贡献的队伍与使用重复代码参赛的队伍。

### 参赛对象

机器学习方向全体学生，并以小组为单位参赛。

### 成绩评定

根据比赛所设评价标准对参赛队伍提交的结果进行成绩评定并排名。比赛完成后将所有队伍分成三组：

**优秀组**：排名靠前6组。奖励包括较大幅度的平时成绩奖励、期末成绩奖励、其它奖励。

**合格组**：奖励包括平时成绩奖励、期末成绩奖励。

**不屈组**：排名靠后6组。无奖励。

### 比赛组织

软件学院机器学习教研室



## 赛题与数据

### 介绍

根据包含3位数字的图像数据来识别图像的内容。例如，当图像中的数字为001时，识别结果为1；当图像中的数字为011时，识别结果为11；当图像中的数字为111时，识别结果为111。

### 数据集

数据集包含以下四个文件：

|文件名称|文件格式|
|:---|:----|
|train.zip|.zip（19.8M）|
|test.zip|.zip（39.8M）|
|train_labels.csv|.csv（95 KB）|
|sample.csv|.csv（86 KB）|

`train.zip`为训练数据集，其中包含了10000张训练图像。`test.zip`为测试数据集，其中包含了20000张测试图像，当模型训练完成之后，需要对测试集进行预测得到预测结果。`train_labels.csv`为训练集标签。`sample.csv`为提交结果的参考模板。

数据集样本说明：每张图像的大小不一定相同，每张图像中包含一个三位数字。图像示例如下：

![example1](./images/50.jpg) 	![example2](./images/53.jpg) 	![example3](./images/70.jpg) 

数据集标签说明：每一行文本代表一个样本标记，每一个整数代表一个图像类别，即图像中显示的数字大小。 

### 提交说明

预测测试集的结果的提交格式参考`sample.csv`中的格式进行提交。要求必须提交`.csv`格式文件，且每条结果与测试集的数据集一一对应。

### 评价指标：

The evaluation metric for this competition is ***Accuracy.***
The Accuracy score is given by:		$Accuracy = (TP+TN) / (TP+TN+FP+FN)$
where: 
-  TP = True  positive  
-  FP = False positive
-  TN = True negative
-   FN = False negative


# 代码
ckpt_cnn.py
```python
# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import random
import time

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 64
MAX_NUM = 3
CHAR_SET_LEN = 10
    
def text2vec(digit):
    text = str(digit)
    vector=np.zeros(CHAR_SET_LEN*MAX_NUM)
    if len(text)==1:
        vector[0]=1
        vector[10]=1
        vector[20+digit]=1
    elif len(text)==2:
        vector[0]=1
        vector[10+digit//10]=1
        vector[20+digit%10]=1
    else:
        vector[digit//100]=1
        vector[10+(digit//10%10)]=1
        vector[20+(digit%10)]=1
    return vector 


    
#生成一个训练batch    
def get_next_batch(batch_size, step, type='train'):
    batch_x = np.zeros([batch_size, IMAGE_WIDTH*IMAGE_HEIGHT])
    batch_y = np.zeros([batch_size, CHAR_SET_LEN*MAX_NUM])
    if type == 'train':
        index = [ i for i in train_text_array]
    elif type == 'valid':
        index = [ i for i in valid_text_array]

#    np.random.shuffle(index)
    totalNumber = len(index) 
    indexStart = step * batch_size   
    for i in range(batch_size):
        idx = index[(i + indexStart) % totalNumber]
        jpg_path = './train/' + str(idx) + '.jpg' 
        img = Image.open(jpg_path)
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
        img = img.convert('L')
        img = np.array(img)
        img = img.flatten() / 255

        text = text_val[idx]
        text = text2vec(text)
        batch_x[i,:] = img
        batch_y[i,:] = text 
    return batch_x, batch_y
    
#构建卷积神经网络并训练
def train_data_with_CNN():
    def weight_variable(shape, name='weight'):
        w_alpha=0.01
#        init = w_alpha*tf.truncated_normal(shape, stddev=0.1)
        init = w_alpha*tf.random_normal(shape)
        var = tf.Variable(initial_value=init, name=name)
        return var
    #初始化偏置    
    def bias_variable(shape, name='bias'):
        b_alpha=0.1
        init = b_alpha * tf.random_normal(shape)
        var = tf.Variable(init, name=name)
        return var
    #卷积    
    def conv2d(x, W, name='conv2d'):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)
    #池化 
    def max_pool_2X2(x, name='maxpool'):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)  

    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH], name='data-input')
    Y = tf.placeholder(tf.float32, [None, MAX_NUM * CHAR_SET_LEN], name='label-input')    
    x_input = tf.reshape(X, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name='x-input')
    #dropout,防止过拟合
    #请注意 keep_prob 的 name，在测试model时会用到它
    keep_prob = tf.placeholder(tf.float32, name='keep-prob')
    #第一层卷积
    W_conv1 = weight_variable([3,3,1,32], 'W_conv1')
    B_conv1 = bias_variable([32], 'B_conv1')
    conv1 = tf.nn.relu(conv2d(x_input, W_conv1, 'conv1') + B_conv1)
    conv1 = max_pool_2X2(conv1, 'conv1-pool')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    #第二层卷积
    W_conv2 = weight_variable([3,3,32,64], 'W_conv2')
    B_conv2 = bias_variable([64], 'B_conv2')
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2,'conv2') + B_conv2)
    conv2 = max_pool_2X2(conv2, 'conv2-pool')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    #第三层卷积
    W_conv3 = weight_variable([3,3,64,64], 'W_conv3')
    B_conv3 = bias_variable([64], 'B_conv3')
    conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 'conv3') + B_conv3)
    conv3 = max_pool_2X2(conv3, 'conv3-pool')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    #全链接层
    #每次池化后，图片的宽度和高度均缩小为原来的一半，进过上面的三次池化，宽度和高度均缩小8倍
    W_fc1 = weight_variable([20*8*64, 1024], 'W_fc1')
    B_fc1 = bias_variable([1024], 'B_fc1')
    fc1 = tf.reshape(conv3, [-1, 20*8*64])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
    fc1 = tf.nn.dropout(fc1, keep_prob)
    #输出层
    W_fc2 = weight_variable([1024, MAX_NUM * CHAR_SET_LEN], 'W_fc2')
    B_fc2 = bias_variable([MAX_NUM * CHAR_SET_LEN], 'B_fc2')
    output = tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output')


    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    optimizer = tf.train.AdamOptimizer(0.0005).minimize(loss)
    
    predict = tf.reshape(output, [-1, MAX_NUM, CHAR_SET_LEN], name='predict')
    labels = tf.reshape(Y, [-1, MAX_NUM, CHAR_SET_LEN], name='labels')
    #预测结果
    #请注意 predict_max_idx 的 name，在测试model时会用到它
    predict_max_idx = tf.argmax(predict, axis=2, name='predict_max_idx')
    labels_max_idx = tf.argmax(labels, axis=2, name='labels_max_idx')
    predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)
    accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps = 0
        ckpt = tf.train.get_checkpoint_state('./models/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        while True:
            train_data, train_label = get_next_batch(32, steps, 'train')
            _,loss_ = sess.run([optimizer,loss], feed_dict={X : train_data, Y : train_label, keep_prob:0.75})
            print("step:%d loss:%f" % (steps,loss_))
            if steps % 100 == 0:
                valid_data, valid_label = get_next_batch(100, steps, 'valid')
                acc = sess.run(accuracy, feed_dict={X : valid_data, Y : valid_label, keep_prob:1.0})
                print("steps=%d, accuracy=%f" % (steps, acc))
                saver.save(sess, "./models/cnn.model", global_step=steps)
                if acc > 0.99:
                    break
            steps += 1


            
if __name__ == '__main__':    
    
    random.seed(time.time())
    #打乱顺序
    label = pd.read_csv('./train_labels.csv')
    text_val = np.array(label['y'])
    text_id = np.array(label['id'])

    TRAIN_SIZE = 0.7
    TRAIN_NUM = int(len(text_id) * TRAIN_SIZE)
    random.shuffle(text_id)
    train_text_array = text_id[:TRAIN_NUM]
    valid_text_array = text_id[TRAIN_NUM:]

    train_data_with_CNN()    
 
```

fit.py
```python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:03:16 2018
@author: zyq
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

label = pd.read_csv('./train_labels.csv')
text_val = np.array(label['y'])
text_id = np.array(label['id'])
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 64
TRAIN_SIZE = 0.8
TRAIN_NUM = int(len(text_id) * TRAIN_SIZE)
train_text_array = text_id[:TRAIN_NUM]
valid_text_array = text_id[TRAIN_NUM:]
MAX_NUM = 3
CHAR_SET_LEN = 10
 

    
def text2vec(digit):
    text = str(digit)
    vector=np.zeros(CHAR_SET_LEN*MAX_NUM)
    if len(text)==1:
        vector[0]=1
        vector[10]=1
        vector[20+digit]=1
    elif len(text)==2:
        vector[0]=1
        vector[10+digit//10]=1
        vector[20+digit%10]=1
    else:
        vector[digit//100]=1
        vector[10+(digit//10%10)]=1
        vector[20+(digit%10)]=1
    return vector 


    
#生成一个训练batch    
def get_next_batch(batch_size, type='train'):
    batch_x = np.zeros([batch_size, IMAGE_WIDTH*IMAGE_HEIGHT])
    batch_y = np.zeros([batch_size, CHAR_SET_LEN*MAX_NUM])
    if type == 'train':
        index = [ i for i in train_text_array]
    elif type == 'valid':
        index = [ i for i in valid_text_array]

    np.random.shuffle(index)

    for i in range(batch_size):
        idx = index[i]
        jpg_path = './train/' + str(idx) + '.jpg' 
        img = Image.open(jpg_path)
        img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR)
        img = img.convert('L')
        img = np.array(img)
        img = img.flatten() / 255

        text = text_val[idx]
        text = text2vec(text)
        batch_x[i,:] = img
        batch_y[i,:] = text 
    return batch_x, batch_y
    
#构建卷积神经网络并训练
def train_data_with_CNN():
    #初始化权值
    def weight_variable(shape, name='weight'):
        w_alpha=0.01
        init = w_alpha*tf.truncated_normal(shape, stddev=0.1)
        # init = w_alpha*tf.random_normal(shape)
        var = tf.Variable(initial_value=init, name=name)
        return var
    #初始化偏置    
    def bias_variable(shape, name='bias'):
        b_alpha=0.1
        init = b_alpha * tf.random_normal(shape)
        var = tf.Variable(init, name=name)
        return var
    #卷积    
    def conv2d(x, W, name='conv2d'):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)
    #池化 
    def max_pool_2X2(x, name='maxpool'):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)     
   
    #输入层
    #请注意 X 的 name，在测试model时会用到它
    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH], name='data-input')
    Y = tf.placeholder(tf.float32, [None, MAX_NUM * CHAR_SET_LEN], name='label-input')    
    x_input = tf.reshape(X, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name='x-input')
    #dropout,防止过拟合
    #请注意 keep_prob 的 name，在测试model时会用到它
    keep_prob = tf.placeholder(tf.float32, name='keep-prob')
    #第一层卷积
    W_conv1 = weight_variable([3,3,1,32], 'W_conv1')
    B_conv1 = bias_variable([32], 'B_conv1')
    conv1 = tf.nn.relu(conv2d(x_input, W_conv1, 'conv1') + B_conv1)
    conv1 = max_pool_2X2(conv1, 'conv1-pool')
    conv1 = tf.nn.dropout(conv1, keep_prob)
    #第二层卷积
    W_conv2 = weight_variable([3,3,32,64], 'W_conv2')
    B_conv2 = bias_variable([64], 'B_conv2')
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2,'conv2') + B_conv2)
    conv2 = max_pool_2X2(conv2, 'conv2-pool')
    conv2 = tf.nn.dropout(conv2, keep_prob)
    #第三层卷积
    W_conv3 = weight_variable([3,3,64,64], 'W_conv3')
    B_conv3 = bias_variable([64], 'B_conv3')
    conv3 = tf.nn.relu(conv2d(conv2, W_conv3, 'conv3') + B_conv3)
    conv3 = max_pool_2X2(conv3, 'conv3-pool')
    conv3 = tf.nn.dropout(conv3, keep_prob)
    #全链接层
    #每次池化后，图片的宽度和高度均缩小为原来的一半，进过上面的三次池化，宽度和高度均缩小8倍
    W_fc1 = weight_variable([12*8*64, 1024], 'W_fc1')
    B_fc1 = bias_variable([1024], 'B_fc1')
    fc1 = tf.reshape(conv3, [-1, 12*8*64])
    fc1 = tf.nn.relu(tf.add(tf.matmul(fc1, W_fc1), B_fc1))
    fc1 = tf.nn.dropout(fc1, keep_prob)
    #输出层
    W_fc2 = weight_variable([1024, MAX_NUM * CHAR_SET_LEN], 'W_fc2')
    B_fc2 = bias_variable([MAX_NUM * CHAR_SET_LEN], 'B_fc2')
    output = tf.add(tf.matmul(fc1, W_fc2), B_fc2, 'output')
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=output))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)
    
    predict = tf.reshape(output, [-1, MAX_NUM, CHAR_SET_LEN], name='predict')
    labels = tf.reshape(Y, [-1, MAX_NUM, CHAR_SET_LEN], name='labels')
    #预测结果
    #请注意 predict_max_idx 的 name，在测试model时会用到它
    predict_max_idx = tf.argmax(predict, axis=2, name='predict_max_idx')
    labels_max_idx = tf.argmax(labels, axis=2, name='labels_max_idx')
    predict_correct_vec = tf.equal(predict_max_idx, labels_max_idx)
    accuracy = tf.reduce_mean(tf.cast(predict_correct_vec, tf.float32))
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        steps = 0
        for epoch in range(6000):
            train_data, train_label = get_next_batch(64, 'train')
            _,loss_ = sess.run([optimizer,loss], feed_dict={X : train_data, Y : train_label, keep_prob:0.75})
            print("step:%d loss_:%f" % (steps,loss_))
            if steps % 100 == 0:
                test_data, test_label = get_next_batch(100, 'valid')
                acc = sess.run(accuracy, feed_dict={X : test_data, Y : test_label, keep_prob:1.0})
                print("steps=%d, accuracy=%f" % (steps, acc))
                saver.save(sess, "./model/crack_captcha.model", global_step=steps)
                if acc > 0.99:
                    break
            steps += 1
            
train_data_with_CNN()
 
```
pre.py
```python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 11:03:52 2018
@author: zyq
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
#import matplotlib.pyplot as plt 
 
CAPTCHA_LEN = 3
IMAGE_WIDTH = 96
IMAGE_HEIGHT = 64
MODEL_SAVE_PATH = './model/'
TEST_IMAGE_PATH = './test/'
 

 
def model_test():
    #加载graph
    saver = tf.train.import_meta_graph(MODEL_SAVE_PATH+"crack_captcha.model-5900.meta")
    graph = tf.get_default_graph()
    #从graph取得 tensor，他们的name是在构建graph时定义的(查看上面第2步里的代码)
    input_holder = graph.get_tensor_by_name("data-input_1:0")
    keep_prob_holder = graph.get_tensor_by_name("keep-prob_1:0")
    predict_max_idx = graph.get_tensor_by_name("predict_max_idx:0")
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_SAVE_PATH))
#        count = 0
        digit_list = []
        for i in range(20000):
            img_path = TEST_IMAGE_PATH + str(i) + '.jpg'
            img = Image.open(img_path)
#            img = img.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.BILINEAR)
            img = img.convert("L")       
            img_array = np.array(img)    
            img_data = img_array.flatten()/255
            
            predict = sess.run(predict_max_idx, feed_dict={input_holder:[img_data], keep_prob_holder : 1.0})            
#            filePathName = img_path
#            print(filePathName)
#            img = Image.open(filePathName)
#            plt.imshow(img)
#            plt.axis('off')
#            plt.show()
            predictValue = np.squeeze(predict)
            digit_list.append(predictValue)
#            print("预测值：{}".format(predictValue))
        return digit_list
        #     if np.array_equal(predictValue, rightValue):
        #         result = '正确'
        #         count += 1
        #     else: 
        #         result = '错误'            
        #     print('实际值：{}， 预测值：{}，测试结果：{}'.format(rightValue, predictValue, result))
        #     print('\n')
            
        # print('正确率：%.2f%%(%d/%d)' % (count*100/totalNumber, count, totalNumber))

def list2digit(y_list):
    arr = []
    for j in range(len(y_list)):
        text = y_list[j]
        digit = 0
        for i,c in enumerate(text):
            digit += pow(10,2-i)*c
        arr.append(digit)
    return arr  
if __name__ == '__main__':
    arr = model_test()
    y_list = list2digit(arr)
    data = pd.DataFrame(y_list)
    data.to_csv('test_labels.csv')
```
