""" 迁移学习模型DAN的实现 """
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import time 
import pandas 
import logging
import os
from datetime import datetime
import random
import argparse
#
from base import MYCNN
from dataload import Data_Path_Load,Load_Batch_Image

#执行 指令设置
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mode', default='train', help='train or test')
#parser.add_argument('--num_epochs', default=1)
#parser.add_argument('--batch_size', default=50)
#parser.add_argument('--learning_rate', default=0.001)
args = parser.parse_args()
#
random.seed(12)

#只使用CPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ='-1' 设置gpu不可见


#设置GPU设备使用
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
#设置GPU使用策略为需要时申请,以改变默认占据全部内存
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu,enable=True)

#tf.config.experimental.set_visible_devices(devices=gpus[0],device_type='GPU')
#tf.config.experimental.set_visible_devices(devices=cpus,device_type='CPU') 

#超参数设置
source_domain_path = "f:/pythonwork/DAN/PACS_dataset/kfold/cartoon/"
target_domain_path = "f:/pythonwork/DAN/PACS_dataset/kfold/photo/"
checkpoint_path = "f:/pythonwork/DAN/savedan"
tb_log_dir = 'f:/pythonwork/DAN/tensorboard'

learning_rate = 0.01
batch_size = 50
n_epoch = 100
drop_rate = 0.4

optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate)

#log
current_path = os.path.abspath('.')
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

#Handler
#file
#把log文件存在当前路径下
log_path = os.path.join(current_path,'log_dan.log')
handler = logging.FileHandler(log_path)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
#console
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s-%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
#
logger.info("----------------------this is log----------------")


""" #model
class MYCNN(tf.keras.Model):
    def __init__(self,rate=0.4):
        super().__init__()
        #原AlexNet框架

        #较少参数
                 
        self.conv1 = tf.keras.layers.Conv2D(
                                        filters=36,
                                        kernel_size = [7,7],
                                        padding='same',
                                        strides=4)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dp1 = tf.keras.layers.Dropout(rate)

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[3,3],strides=2,padding='valid') 
        self.conv2 = tf.keras.layers.Conv2D(
                                        filters=54,
                                        kernel_size = [5,5],
                                        padding='same',
                                        strides=3)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dp2 = tf.keras.layers.Dropout(rate)

        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[3,3],strides=2,padding='valid')
        self.conv3 = tf.keras.layers.Conv2D(
                                        filters=18,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=1)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dp3 = tf.keras.layers.Dropout(rate)

        self.conv4 = tf.keras.layers.Conv2D(
                                        filters=18,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=1)
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.dp4 = tf.keras.layers.Dropout(rate)

        self.conv5 = tf.keras.layers.Conv2D(
                                        filters=5,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=1)
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.dp5 = tf.keras.layers.Dropout(rate)

        self.flatten = tf.keras.layers.Flatten()

        self.dense_ada8 = tf.keras.layers.Dense(units=36)
        self.bn6 = tf.keras.layers.BatchNormalization()
        self.dp6 = tf.keras.layers.Dropout(rate)

        self.dense_ada9 = tf.keras.layers.Dense(units=18)
        self.bn7 = tf.keras.layers.BatchNormalization()
        self.dp7 = tf.keras.layers.Dropout(rate)

        self.dense_ada10 = tf.keras.layers.Dense(units=7) 

        #其它参数
        self.training = True
        #当处于训练状态时，默认inputs包含source，target domain同等数量的数据。

        #不能在类成员函数的参数里带self.******，任何self.的变量都不行
    def call(self,inputs):
        training = self.training

        x = self.conv1(inputs)
        x = self.bn1(x,training=training)
        x = tf.nn.relu(x)
        x = self.dp1(x,training=training)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x,training=training)
        x = tf.nn.relu(x)
        x = self.dp2(x,training=training)

        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x,training=training)
        x = tf.nn.relu(x)
        x = self.dp3(x,training=training)

        x = self.conv4(x)
        x = self.bn4(x,training=training)
        x = tf.nn.relu(x)
        x = self.dp4(x,training=training)

        x = self.conv5(x)
        x = self.bn5(x,training=training)
        x = tf.nn.relu(x)
        x = self.dp5(x,training=training)


        x = self.flatten(x)

        x = self.dense_ada8(x)
        x = self.bn6(x,training=training)
        x = tf.nn.relu(x)
        x = self.dp6(x,training=training)

        x = self.dense_ada9(x)
        x = self.bn7(x,training=training)
        x = tf.nn.relu(x)
        x = self.dp7(x,training=training)

        logits = self.dense_ada10(x)
        
        outputs = tf.nn.softmax(logits)
        
        return outputs """

#==========================数据加载，处理==============================================
#数据处理


source_load = Data_Path_Load(source_domain_path)
target_load = Data_Path_Load(target_domain_path)
#不同定义域训练测试数据分割比例[0,1]
source_split_ratio = 0.8 
target_split_ratio = 0.4 

s_train_paths,s_train_labels,s_test_paths,s_test_labels = source_load.get_train_test_set(split_ratio = source_split_ratio)
t_train_paths,t_train_labels,t_test_paths,t_test_labels = target_load.get_train_test_set(split_ratio = target_split_ratio)


#==============================训练设置=====================================
model = MYCNN(rate=drop_rate)   

#训练可视化tensorboard
summary_writer = tf.summary.create_file_writer(tb_log_dir) #保存在当前目录下


#依据paths 按批次加载图片
batch_image_load = Load_Batch_Image()

#批次划分，依赖Load_Batch_Image
class SourceBatch(object):
    def __init__(self):
        super().__init__()

    def shuff_batch(self,X,y,batch_size):
        #tensor list不支持数组切片的方式，要转成np.array 这步非常耗时
        #X =np.array(X)
        #y =np.array(y)
        #将源域数据划分批次
        '''
        生成器
        args:   X为所有data的paths,
                y为对应的label
        return: 混乱后的batch images与对应label
        '''
        rnd_idx = list(np.random.permutation(len(X)))
        n_batch = len(X)//batch_size
        #for idx_batch in np.array_split(rnd_idx,n_batch):  #第一组竟然会切出batch_size+1个
        for idx_batch in self.list_split_even(rnd_idx,n_batch,batch_size):  
            X_batch,y_batch = [],[]
            for idx in idx_batch:
                X_batch.append(X[idx])
                y_batch.append(y[idx])
        
            yield batch_image_load.get_batch_images(X_batch),y_batch

    def list_split_even(self,alist,n_batch,batch_size):
        #对随机指标按num分割，并保证最后一批长度为偶数(文章中的要求)
        idx_batchs = []
        if (len(alist)%batch_size)%2 == 1:
            alist.pop()

        for i in range(n_batch):
            idx_batchs.append(alist[i*batch_size:(i+1)*batch_size])
        idx_batchs.append(alist[n_batch*batch_size:])

        return idx_batchs

class TargetBatch(object):
    def __init__(self,X,y):
        '''
        依赖 Load_Batch_image类
        args:   X为image paths
                y为对应 label
        '''
        self.rest_data = X
        self.rest_labels = y
        self.orgin_data = X
        self.orgin_labels = y

    def get_batch(self,length):
        '''
        return:与length等长的data,label
        '''
        if len(self.rest_labels)<length:
            self.rest_data = self.orgin_data
            self.rest_labels = self.orgin_labels

        if len(self.rest_labels) == length:
            X = self.rest_data
            y = self.rest_labels
        else:
            X = self.rest_data[:length]
            y = self.rest_labels[:length]

            self.rest_data = self.rest_data[length:]
            self.rest_labels = self.rest_labels[length:]


        return batch_image_load.get_batch_images(X),y
    

#保证target domian data 可以循环使用的批次划分,每批和源域数据每批样本量相同。

source_batch = SourceBatch()
target_batch = TargetBatch(t_train_paths,t_train_labels)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
#========================训练=============================================
""" 
#@tf.function
def train_step(sX_batch,tX_batch,sy_batch):
    #只对部分target data作为有label求交叉熵
    #当前为无监督，target data全部无label

    st_batch = tf.concat([sX_batch,tX_batch],axis=0)    #将数据拼接并转为tensor
    with tf.GradientTape() as tape:
        output_batch,mmd = model(st_batch)

        pred_batch = output_batch[:len(sy_batch)]        #只用源域上的label 算结构误差
        loss_batch = loss_object(sy_batch,pred_batch)    #自动的将batch中每个的loss加和平均
        loss = loss_batch + np.sum(lamda*mmd)

    gradient = tape.gradient(loss,model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradient,model.variables))

    train_loss(loss)
    train_accuracy(sy_batch, pred_batch)
"""
#@tf.function
def test_step(model,images, labels):
    model.training = False
    test_loss.reset_states()
    test_accuracy.reset_states()

    predictions = model(tf.convert_to_tensor(images))
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

    model.training = True

#cnn train step
def train_step(sX_batch,sy_batch):
    #只对部分target data作为有label求交叉熵
    #当前为无监督，target data全部无label

    st_batch = tf.convert_to_tensor(sX_batch)
    with tf.GradientTape() as tape:
        pred_batch = model(st_batch)         
        loss = loss_object(sy_batch,pred_batch)    #自动的将batch中每个的loss加和平均
         

    gradient = tape.gradient(loss,model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradient,model.variables))

    train_loss(loss)
    train_accuracy(sy_batch, pred_batch)

def train():
    #保存点
    checkpoint = tf.train.Checkpoint(DANmodel = model)
    manager_ckpt = tf.train.CheckpointManager(
                            checkpoint,
                            directory=checkpoint_path,
                            checkpoint_name='dan_model.ckpt',
                            max_to_keep=3
                            )
    #开始迭代
    logger.info('------------train start----------------------------')
    step = 0
    #tf.summary.trace_on(graph=True,profiler=True) # 开启Trace，可以记录图结构和profile信息
    start_time = time.time()
    for epoch in range(n_epoch):
        #每轮清除缓存
        train_loss.reset_states()
        train_accuracy.reset_states()
        

        for sX_batch,sy_batch in source_batch.shuff_batch(s_train_paths,s_train_labels,batch_size):
            #保证与当次source batch 内样本数目相同
            #tX_batch,ty_batch = target_batch.get_batch(len(sy_batch))

            #执行
            train_step(sX_batch,sy_batch)
        
            step += 1
            if step%5 == 0:
                #可视化写入
                with summary_writer.as_default():
                    tf.summary.scalar('loss',train_loss.result(),step=step)
                    tf.summary.scalar('accuracy',train_accuracy.result(),step=step)

        if epoch%5 == 0:
            #保存模型
            temp_path = manager_ckpt.save(checkpoint_number=epoch)
            print("model saved to %s"% temp_path)

    
        template = 'Epoch {}, Loss: {},train Accuracy: {}'
        print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100)
                        )
        logger.info(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100)
                        )
    #训练完，保存
    temp_path = manager_ckpt.save(checkpoint_number=epoch)
    print("model saved to %s"% temp_path)

    logger.info('------------train finish----------------------------')

    end_time = time.time()
    print("spend total time:{} train sample num: {}".format(end_time-start_time,len(s_train_labels))) 
    logger.info("spend total time:{} train sample num: {}".format(end_time-start_time,len(s_train_labels)))
    logger.info(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100)
                        )

def test():
    logger.info('------------test start----------------------------')

    #模型恢复
    model_restore = MYCNN()
    checkpoint = tf.train.Checkpoint(DANmodel = model_restore)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

    sX_test = batch_image_load.get_batch_images(s_test_paths)

    test_step(model_restore,sX_test,s_test_labels)

    template = 'source test num: {},test set Loss: {},test Accuracy: {}'
    print(template.format(len(s_test_labels),
                        test_loss.result(),
                        test_accuracy.result()*100)
                        )
    logger.info(template.format(len(s_test_labels),
                        test_loss.result(),
                        test_accuracy.result()*100)
                        )
    #测试target域的结果
    tX_test = batch_image_load.get_batch_images(t_test_paths)
    test_step(model_restore,tX_test,t_test_labels)

    template = 'target test num: {},test set Loss: {},test Accuracy: {}'
    print(template.format(len(t_test_labels),
                        test_loss.result(),
                        test_accuracy.result()*100)
                        )
    logger.info(template.format(len(t_test_labels),
                        test_loss.result(),
                        test_accuracy.result()*100)
                        )

    logger.info('------------test finish----------------------------')


if __name__ == '__main__':
    #在命令行参数中加入 --mode=[test /train]
    if args.mode == "train":
        train()
    if args.mode == "test":
        test()

logger.info('------------------logging finish---------------------')