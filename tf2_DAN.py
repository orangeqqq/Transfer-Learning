""" 迁移学习模型DAN的实现 """
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt 
import time 
import pandas 
import logging
import functools

#log
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

#Handler
#file
handler = logging.FileHandler('f:/pythonwork/Transfer-Learning/dan.log')
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
logger.info("this is log")

#自适应层
class AdaptiveLayer(tf.keras.layers.Layer):
    def __init__(self,units,activation=None,gamma=1):
        super().__init__()
        # 初始化代码
        self.units = units
        self.activation = activation
        self.gamma = gamma

    def build(self, input_shape):
        # input_shape 是一个 TensorShape 类型对象，提供输入的形状
        # 在第一次使用该层的时候调用该部分代码，在这里创建变量可以使得变量的形状自适应输入的形状
        # 而不需要使用者额外指定变量形状。
        # 如果已经可以完全确定变量的形状，也可以在__init__部分创建变量
        self.w = self.add_weight(name='kernel',shape=[input_shape[-1],self.units],initializer=tf.zeros_initializer())
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
            domain_loss = mk_MMD(domain[0],domain[1])

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
            distance += g_k(domain1_inputs[i],domain1_inputs[i+1],
                domain2_inputs[i],domain2_inputs[i+1])
            i +=2

        distance = distance*2/num_s

        return distance



    def g_k(self,xs_i,xs_j,xt_i,xt_j):
        y = m_PSDkernels(xs_i,xs_j) + m_PSDkernels(xt_i,xt_j) \
            - m_PSDkernels(xs_i,xt_j) - m_PSDkernels(xs_j,xt_i)

        return y 

    #多重核的凸包
    def m_PSDkernels(self,x_i,x_j,beta=8):
        m_beta = tf.ones([beta])/beta
        gamma_u = self.gamma/(2**(beta//2))
        convex_sum = tf.constant(0,dtype=tf.float32)

        for beta_u in m_beta:
            convex_sum += beta_u * kernel_u(x_i,x_j,gamma_u)
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
        self.conv1 = tf.keras.layers.Conv2D(activation=tf.nn.relu,
                                        filters=6,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=4)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[3,3],strides=2,padding='valid') 
        self.conv2 = tf.keras.layers.Conv2D(activation=tf.nn.relu,
                                        filters=6,
                                        kernel_size = [5,5],
                                        padding='same',
                                        strides=1)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[3,3],strides=2,padding='valid')
        self.conv3 = tf.keras.layers.Conv2D(activation=tf.nn.relu,
                                        filters=4,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=1)
        self.conv4 = tf.keras.layers.Conv2D(activation=tf.nn.relu,
                                        filters=4,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=1)
        self.conv5 = tf.keras.layers.Conv2D(activation=tf.nn.relu,
                                        filters=2,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense_ada8 = AdaptiveLayer(units=36,activation=tf.nn.relu)
        self.dense_ada9 = AdaptiveLayer(units=36,activation=tf.nn.relu)
        self.dense_ada10 = AdaptiveLayer(units=7)

        #其它参数
        self.training = True
        #当处于训练状态时，默认inputs包含source，target domain同等数量的数据。

        #不能在类成员函数的参数里带self.******，任何self.的变量都不行
    def call(self,inputs):
        training = self.training

        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        if training:
            x,mmk_1 = self.dense_ada8(x,training=training)
            x,mmk_2 = self.dense_ada9(x,training=training)
            logits,mmk_3 = self.dense_ada10(x,training=training)
            mmk = [mmk_1,mmk_2,mmk_3]
            outputs = tf.nn.softmax(logits)

            return outputs,mmk
        else:
            x = self.dense_ada8(x,training=training)
            x = self.dense_ada9(x,training=training)
            logits = self.dense_ada10(x,training=training)
        
        outputs = tf.nn.softmax(logits)
        
        return outputs

#==========================数据加载，处理==============================================
import pathlib
import random

class DataLoad(object):
    def __init__(self,domain_path):
        self.domain_path = pathlib.Path(domain_path)
        
    def get_train_test_set(self,split_ratio):

        all_image_paths = list(self.domain_path.glob('*/*'))
        all_image_paths = [str(path) for path in all_image_paths]
        self.image_count = len(all_image_paths)

        label_names = sorted(item.name for item in self.domain_path.glob('*/') if item.is_dir())
        label_to_index = dict((name, index) for index, name in enumerate(label_names))

        all_image_labels = [label_to_index[pathlib.Path(path).parent.name] 
                   for path in all_image_paths]
        
        train_paths,train_labels,test_paths,test_labels = \
            self.shuffle_and_split_path_label(all_image_paths,all_image_labels,split_ratio)
        try:
            train_data = list(map(self.load_and_preprocess_image,train_paths))
            test_data = list(map(self.load_and_preprocess_image,test_paths))
        except Exception as e:
            logger.error('Error',exc_info=True)
            raise e


        return train_data,train_labels,test_data,test_labels 

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
    def preprocess_image(self,image):
        image = tf.image.decode_jpeg(image,channels=3)
        image = tf.image.resize(image,[227,227])
        image /=255
    
        return image



    def load_and_preprocess_image(self,path):
        image = tf.io.read_file(path)
    
        return self.preprocess_image(image)
#数据处理
source_path = "f:/Dataset/PACS_dataset/kfold/cartoon/"
source_load = DataLoad(source_path)

target_path = "f:/Dataset/PACS_dataset/kfold/photo/"
target_load = DataLoad(source_path)



source_data,source_labels,_,_ = source_load.get_train_test_set(split_ratio = 1.0)
train_data,train_labels,test_data,test_labels = target_load.get_train_test_set(split_ratio = 0.5)


#==============================训练=====================================
learning_rate = 0.01
batch_size = 50
n_epoch = 30
lamda = [0.5,0.5,1.]

optimizer = tf.keras.optimizers.Adam(learning_rate= learning_rate)

model = DAN()
accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

def shuff_batch(X,y,batch_size):
    #tensor 不支持数组切片的方式，要转成np.array
    X =np.array(X)
    y =np.array(y)

    rnd_idx = np.random.permutation(len(X))
    n_batch = len(X)//batch_size
    for idx_batch in np.array_split(rnd_idx,n_batch):
        X_batch,y_batch = X[idx_batch],y[idx_batch]
        yield X_batch,y_batch

class TargetBatch(object):
    def __init__(self,X,y):
        self.rest_data = X
        self.rest_labels = y
        self.orgin_data = X
        self.orgin_labels = y

    def get_batch(self,length):
        if len(self.rest_labels)<length:
            self.rest_data = self.orgin_data
            self.rest_labels = self.orgin_labels

        if len(self.rest_labels) == length:
             X = self.rest_data
             y = self.rest_labels
        else:
             X = self.rest_data[:length]
             y = self.rest_labels[:length]

        return X,y

target_batch = TargetBatch(train_data,train_labels)

for epoch in range(n_epoch):
    for sX_batch,sy_batch in shuff_batch(source_data,source_labels,batch_size):
        #保证与当次source batch 内样本数目相同
        tX_batch,ty_batch = target_batch.get_batch(len(sy_batch))
        st_batch = tf.concat([sX_batch,tX_batch],axis=0)
               
        with tf.GradientTape() as tape:
            output_batch,mmd = model(st_batch)
            #只对部分target data作为有label求交叉熵
            #当前为无监督，targetdata全部无label
            pred_batch = output_batch[:len(sy_batch)]
            loss_batch = tf.keras.losses.sparse_categorical_crossentropy(sy_batch,pred_batch,from_logits= False) 
            loss = tf.reduce_mean(loss_batch) + tf.sum(lamda*mmd)

        gradient = tape.gradient(loss,model.variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradient,model.variables))

    if epoch%2 == 0:
        y_pred = model.predict(X_batch)
        acc = accuracy(y_batch,y_pred)
        print("epoch %d train accuracy is %f"%(epoch,acc))

#评估
model.training = False 

y_pred = model.predict(test_data)
acc = accuracy(test_labels,y_pred)
print("Test set accuracy is %f"%acc)



logger.info('logging finish')

