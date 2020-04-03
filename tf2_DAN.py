""" 迁移学习模型DAN的实现 """
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import time 
import pandas 
import logging
import functools
import os
from datetime import datetime
import pathlib
import random

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

tf.config.experimental.set_visible_devices(devices=gpus[0],device_type='GPU')
#tf.config.experimental.set_visible_devices(devices=cpus,device_type='CPU')

#超参数设置
source_domain_path = "f:/pythonwork/DAN/PACS_dataset/kfold/cartoon/"
target_domain_path = "f:/pythonwork/DAN/PACS_dataset/kfold/photo/"
checkpoint_path = "f:/pythonwork/DAN/save_dan"
tb_log_dir = 'f:/pythonwork/DAN/tensorboard'

learning_rate = 0.01
batch_size = 50
n_epoch = 15
lamda = np.array([1,1,1])  #作者实验给出

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

#自适应层
class AdaptiveLayer(tf.keras.layers.Layer):
    def __init__(self,units,activation=None,gamma=1):
        super().__init__()
        # 初始化代码
        self.units = units
        self.activation = activation
        self.gamma = gamma             #gammam用于选择核函数

    def build(self, input_shape):
        # input_shape 是一个 TensorShape 类型对象，提供输入的形状
        # 在第一次使用该层的时候调用该部分代码，在这里创建变量可以使得变量的形状自适应输入的形状
        # 而不需要使用者额外指定变量形状。
        # 如果已经可以完全确定变量的形状，也可以在__init__部分创建变量
        self.w = self.add_weight(name='kernel',shape=[input_shape[-1],self.units],initializer=tf.zeros_initializer())  #因为是全连接层，所以权重shape 应取input最后一维
        self.b = self.add_weight(name="bias",shape=[self.units],initializer=tf.zeros_initializer())

    """def call(self, inputs_1,inputs_2=None,training=False):
        # 模型调用的代码（处理输入并返回输出）
        #有多个输入时，tf会自动将第一个输入的shape赋给input_shape。保证多个输入的shape一致即可。
        if training:
            inputs_list = [inputs_1,inputs_2]
        else:
            inputs_list = [inputs_1]

        domain = []
        for inputs in inputs_list: 
            x = tf.matmul(inputs,self.w) + self.b
            if self.activation not is  None :
                x = self.activation(x)
            domain.append(x)

        if training:
            domain_loss = mk_MMD(domain[0],domain[1])
             """
    def call(self, inputs,training=False):
        # 模型调用的代码（处理输入并返回输出）
        #有多个输入时，tf会自动将第一个输入的shape赋给input_shape。保证多个输入的shape一致即可。

        
        outputs = tf.matmul(inputs,self.w) + self.b
        if self.activation != None :
            outputs = self.activation(outputs)
        

        if training:
            split_index = len(outputs)//2
            domain = [outputs[:split_index],outputs[split_index:]]
            domain_loss = self.mk_MMD(domain[0],domain[1])

            return outputs,domain_loss
        
        return outputs
    
    def mk_MMD(self,domain1_inputs,domain2_inputs):
        if domain1_inputs.shape[0]%2 != 0 or domain2_inputs.shape[0]%2 != 0:
            logger.info("adaption layers error")
            raise Exception("all domain  data batch size must be times of 2 !")
        elif len(domain1_inputs) != len(domain2_inputs):
            logger.info("adaption layers error")
            raise Exception("source domain and target domain  batch size must equal !")

        num_s = len(domain1_inputs)
        i = 0
        distance = tf.constant(0,dtype=tf.float32)
        while i < num_s :
            distance += self.g_k(domain1_inputs[i],domain1_inputs[i+1],
                domain2_inputs[i],domain2_inputs[i+1])
            i +=2

        distance = distance*2/num_s

        return distance



    def g_k(self,xs_i,xs_j,xt_i,xt_j):
        y = self.m_PSDkernels(xs_i,xs_j) + self.m_PSDkernels(xt_i,xt_j) \
            - self.m_PSDkernels(xs_i,xt_j) - self.m_PSDkernels(xs_j,xt_i)

        return y 

    #多重核的凸包
    def m_PSDkernels(self,x_i,x_j,beta=8):
        m_beta = tf.ones([beta])/beta
        gamma_u = self.gamma/(2**(beta//2))
        convex_sum = tf.constant(0,dtype=tf.float32)

        for beta_u in m_beta:
            convex_sum += beta_u * self.kernel_u(x_i,x_j,gamma_u)
            gamma_u = gamma_u*2

        return convex_sum

    def kernel_u(self,x_i,x_j,gamma_u):

        y = tf.math.square(tf.norm(x_i-x_j))/gamma_u
        return tf.math.exp(-y)

#model
class DAN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        #原AlexNet框架
        """          
        self.conv1 = tf.keras.layers.Conv2D(activation=tf.nn.relu,
                                        filters=96,
                                        kernel_size = [11,11],
                                        padding='same',
                                        strides=4)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[3,3],strides=2,padding='valid') 
        self.conv2 = tf.keras.layers.Conv2D(activation=tf.nn.relu,
                                        filters=256,
                                        kernel_size = [5,5],
                                        padding='same',
                                        strides=1)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[3,3],strides=2,padding='valid')
        self.conv3 = tf.keras.layers.Conv2D(activation=tf.nn.relu,
                                        filters=384,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=1)
        self.conv4 = tf.keras.layers.Conv2D(activation=tf.nn.relu,
                                        filters=384,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=1)
        self.conv5 = tf.keras.layers.Conv2D(activation=tf.nn.relu,
                                        filters=256,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_ada8 = AdaptiveLayer(units=4096,activation=tf.nn.relu)
        self.dense_ada9 = AdaptiveLayer(units=4096,activation=tf.nn.relu)
        self.dense_ada10 = AdaptiveLayer(units=7) 
        """

        #较少参数
                 
        self.conv1 = tf.keras.layers.Conv2D(
                                        filters=54,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=4)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[3,3],strides=2,padding='valid') 
        self.conv2 = tf.keras.layers.Conv2D(
                                        filters=36,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=3)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[3,3],strides=2,padding='valid')
        self.conv3 = tf.keras.layers.Conv2D(
                                        filters=18,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=2)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.conv4 = tf.keras.layers.Conv2D(
                                        filters=9,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=1)
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.conv5 = tf.keras.layers.Conv2D(
                                        filters=5,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=1)
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.dense_ada8 = AdaptiveLayer(units=36,activation=tf.nn.relu)
        #self.bn6 = tf.keras.layers.BatchNormalization()
        self.dense_ada9 = AdaptiveLayer(units=18,activation=tf.nn.relu)
        self.dense_ada10 = AdaptiveLayer(units=7) 

        #其它参数
        self.training = True
        #当处于训练状态时，默认inputs包含source，target domain同等数量的数据。

        #不能在类成员函数的参数里带self.******，任何self.的变量都不行
    def call(self,inputs):
        training = self.training

        x = self.conv1(inputs)
        x = self.bn1(x,training=training)
        x = tf.nn.relu(x)

        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x,training=training)
        x = tf.nn.relu(x)

        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x,training=training)
        x = tf.nn.relu(x)

        x = self.conv4(x)
        x = self.bn4(x,training=training)
        x = tf.nn.relu(x)

        x = self.conv5(x)
        x = self.bn5(x,training=training)
        x = tf.nn.relu(x)

        x = self.flatten(x)
        if training:
            x,mmk_1 = self.dense_ada8(x,training=training)
            x,mmk_2 = self.dense_ada9(x,training=training)
            logits,mmk_3 = self.dense_ada10(x,training=training)
            mmk = np.array([mmk_1,mmk_2,mmk_3])
            outputs = tf.nn.softmax(logits)

            return outputs,mmk
        else:
            x = self.dense_ada8(x,training=training)
            x = self.dense_ada9(x,training=training)
            logits = self.dense_ada10(x,training=training)
        
        outputs = tf.nn.softmax(logits)
        
        return outputs

#==========================数据加载，处理==============================================

class Data_Path_Load(object):
    '''
    从目标文件夹（domain）加载数据path，并创建对应的label.
    将数据打乱顺序返回。
    '''
    def __init__(self,domain_path):
        self.domain_path = pathlib.Path(domain_path)
        
    def get_train_test_set(self,split_ratio):
        '''

        '''
        all_image_paths = list(self.domain_path.glob('*/*'))
        all_image_paths = [str(path) for path in all_image_paths]
        self.image_count = len(all_image_paths)

        label_names = sorted(item.name for item in self.domain_path.glob('*/') if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(label_names))

        all_image_labels = [label_to_index[pathlib.Path(path).parent.name] 
                   for path in all_image_paths]
        
        train_paths,train_labels,test_paths,test_labels = \
            self.shuffle_and_split_path_label(all_image_paths,all_image_labels,split_ratio)
        """             
        try:
            train_data = list(map(self.load_and_preprocess_image,train_paths))
            test_data = list(map(self.load_and_preprocess_image,test_paths))
        except Exception as e:
            logger.error('Error',exc_info=True)
            raise e 
        """


        return train_paths,train_labels,test_paths,test_labels 

    #混淆数据顺序
    def shuffle_and_split_path_label(self,all_image_paths,all_image_labels,split_ratio):

        if split_ratio > 1.0 or split_ratio < 0 :
            raise ValueError("data split should be in [0,1]")
        
        image_label_zip = list(zip(all_image_paths,all_image_labels))
        random.shuffle(image_label_zip)
        all_image_paths[:],all_image_labels[:] = zip(*image_label_zip)

        if split_ratio == 1.:
            
            train_paths,test_paths = all_image_paths,[]
            train_labels,test_labels = all_image_labels,[]
        elif split_ratio == 0.:
            train_paths,test_paths = [],all_image_paths
            train_labels,test_labels = [],all_image_labels
        else:

            split_index = int(self.image_count * split_ratio)
            train_paths,test_paths = all_image_paths[:split_index],all_image_paths[split_index:]
            train_labels,test_labels = all_image_labels[:split_index],all_image_labels[split_index:]

        return train_paths,train_labels,test_paths,test_labels

#加载图像函数
class Load_Batch_Image(object):
    def __init__(self):
        super().__init__()

    def get_batch_images(self,image_paths):
        '''
        return: 返回list,每个分量为tensor
        '''        
        try:
            batch_images = list(map(self.load_and_preprocess_image,image_paths))
        except Exception as e:
            logger.error('Error',exc_info=True)
            raise e 

        return batch_images
                

    def preprocess_image(self,image):
        image = tf.image.decode_jpeg(image,channels=3)
        image = tf.image.resize(image,[227,227])
        image /=255
    
        return image

    def load_and_preprocess_image(self,path):
        image = tf.io.read_file(path)
    
        return self.preprocess_image(image)
#数据处理


source_load = Data_Path_Load(source_domain_path)
target_load = Data_Path_Load(target_domain_path)
#不同定义域训练测试数据分割比例[0,1]
source_split_ratio = 1.0 
target_split_ratio = 0.6 

source_paths,source_labels,_,_ = source_load.get_train_test_set(split_ratio = source_split_ratio)
train_paths,train_labels,test_paths,test_labels = target_load.get_train_test_set(split_ratio = target_split_ratio)


#==============================训练设置=====================================

model = DAN()
#accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
#保存点
checkpoint = tf.train.Checkpoint(DANmodel = model)
manager_ckpt = tf.train.CheckpointManager(
                            checkpoint,
                            directory=checkpoint_path,
                            checkpoint_name='dan_model.ckpt',
                            max_to_keep=3
                            )

#训练可视化tensorboard
summary_writer = tf.summary.create_file_writer(tb_log_dir) #保存在当前目录下


#依据paths 按批次加载图片
batch_image_load = Load_Batch_Image()

def shuff_batch(X,y,batch_size):
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
    for idx_batch in list_split_even(rnd_idx,n_batch,batch_size):  
        X_batch,y_batch = [],[]
        for idx in idx_batch:
            X_batch.append(X[idx])
            y_batch.append(y[idx])
        
        yield batch_image_load.get_batch_images(X_batch),y_batch

def list_split_even(alist,n_batch,batch_size):
    #对随机指标按num分割，并保证最后一批长度为偶数(文章中的要求)
    idx_batchs = []
    if (len(alist)%batch_size)%2 == 1:
        alist.pop()

    for i in range(n_batch):
        idx_batchs.append(alist[i*batch_size:(i+1)*batch_size])
    idx_batchs.append(alist[n_batch*batch_size:])

    return idx_batchs
    

#保证target domian data 可以循环使用的批次划分,每批和源域数据每批样本量相同。
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

target_batch = TargetBatch(train_paths,train_labels)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
#========================训练=============================================
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

#@tf.function
def test_step(images, labels):
    model.training = False

    predictions = model(tf.convert_to_tensor(images))
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

    model.training = True

#开始迭代
step = 0
#tf.summary.trace_on(graph=True,profiler=True) # 开启Trace，可以记录图结构和profile信息
start_time = time.time()
for epoch in range(n_epoch):
    #每轮清除缓存
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    for sX_batch,sy_batch in shuff_batch(source_paths,source_labels,batch_size):
        #保证与当次source batch 内样本数目相同
        tX_batch,ty_batch = target_batch.get_batch(len(sy_batch))
        #执行
        train_step(sX_batch,tX_batch,sy_batch)
        
        step += 1
        if step%5 == 0:
            #可视化写入
            with summary_writer.as_default():
                tf.summary.scalar('loss',train_loss.result(),step=step)
                tf.summary.scalar('accuracy',train_accuracy.result(),step=step)
    """             
    if epoch == 0:
        #保存trace
        with summary_writer.as_default():
            tf.summary.trace_export(name='DAN_trace',step=step,
                        profiler_outdir=os.path.join(tb_log_dir,datetime.now().strftime("%Y%m%d-%H%M%S")))
    """

    if epoch%5 == 0:
        #保存模型
        temp_path = manager_ckpt.save(checkpoint_number=epoch)
        print("model saved to %s"% temp_path)

    """  
    test_step(batch_image_load.get_batch_images(test_paths),test_labels)
        
    
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100)) 
    """
    template = 'Epoch {}, Loss: {},train Accuracy: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100)
                        )

#训练完，保存
temp_path = manager_ckpt.save(checkpoint_number=epoch)
print("model saved to %s"% temp_path)

logger.info('------------train finish----------------------------')

end_time = time.time()
print("spend total time:%f"%(end_time-start_time)) 
logger.info("spend total time:%f"%(end_time-start_time))
logger.info(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100)
                        )
""" 
test_step(batch_image_load.get_batch_images(test_paths),test_labels)
print('test numbers: {}, test loss: {}, test Accuracy:{}'.format(len(test_labels),
                                test_loss.result(),
                                test_accuracy.result()*100
                                ))
"""
logger.info('------------------logging finish---------------------')

