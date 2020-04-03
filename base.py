import tensorflow as tf
import numpy as np 

#cnn model
class MYCNN(tf.keras.Model):
    def __init__(self,rate=0.4):
        super().__init__()
        #原AlexNet框架

        #较少参数
                 
        self.conv1 = tf.keras.layers.Conv2D(
                                        filters=32,
                                        kernel_size = [9,9],
                                        padding='same',
                                        strides=4)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dp1 = tf.keras.layers.Dropout(rate)

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[3,3],strides=2,padding='valid') 
        self.conv2 = tf.keras.layers.Conv2D(
                                        filters=32,
                                        kernel_size = [5,5],
                                        padding='same',
                                        strides=3)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dp2 = tf.keras.layers.Dropout(rate)

        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[3,3],strides=2,padding='valid')
        self.conv3 = tf.keras.layers.Conv2D(
                                        filters=24,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=1)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dp3 = tf.keras.layers.Dropout(rate)

        self.conv4 = tf.keras.layers.Conv2D(
                                        filters=16,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=1)
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.dp4 = tf.keras.layers.Dropout(rate)

        self.conv5 = tf.keras.layers.Conv2D(
                                        filters=8,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=1)
        self.bn5 = tf.keras.layers.BatchNormalization()
        self.dp5 = tf.keras.layers.Dropout(rate)

        self.flatten = tf.keras.layers.Flatten()

        self.dense_ada8 = tf.keras.layers.Dense(units=32)
        self.bn6 = tf.keras.layers.BatchNormalization()
        self.dp6 = tf.keras.layers.Dropout(rate)

        self.dense_ada9 = tf.keras.layers.Dense(units=16)
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
        
        return outputs

""" #model v2
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

""" #v1 cnn  Trainable params: 42,538
class MYCNN(tf.keras.Model):
    def __init__(self,rate=0.4):
        super().__init__()
        #原AlexNet框架

        #较少参数
                 
        self.conv1 = tf.keras.layers.Conv2D(
                                        filters=54,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=3)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dp1 = tf.keras.layers.Dropout(rate)

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[3,3],strides=2,padding='valid') 
        self.conv2 = tf.keras.layers.Conv2D(
                                        filters=36,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=1)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.dp2 = tf.keras.layers.Dropout(rate)

        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[3,3],strides=2,padding='valid')
        self.conv3 = tf.keras.layers.Conv2D(
                                        filters=18,
                                        kernel_size = [3,3],
                                        padding='same',
                                        strides=2)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.dp3 = tf.keras.layers.Dropout(rate)

        self.conv4 = tf.keras.layers.Conv2D(
                                        filters=9,
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

#dan model
class DAN(tf.keras.Model):
    def __init__(self,basemodel):
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
        # 迁移的卷积层  包含 17 层
        self.conv1 = basemodel.get_layer(index=0)
        self.bn1 = basemodel.get_layer(index=1)
        self.dp1 = basemodel.get_layer(index=2)

        self.pool1 = basemodel.get_layer(index=3) 
        self.conv2 = basemodel.get_layer(index=4)
        self.bn2 = basemodel.get_layer(index=5)
        self.dp2 = basemodel.get_layer(index=6)

        self.pool2 = basemodel.get_layer(index=7)
        self.conv3 = basemodel.get_layer(index=8)
        self.bn3 = basemodel.get_layer(index=9)
        self.dp3 = basemodel.get_layer(index=10)

        self.conv4 = basemodel.get_layer(index=11)
        self.bn4 = basemodel.get_layer(index=12)
        self.dp4 = basemodel.get_layer(index=13)

        self.conv5 = basemodel.get_layer(index=14)
        self.bn5 = basemodel.get_layer(index=15)
        self.dp5 = basemodel.get_layer(index=16)

        #模型特异化的部分
        self.flatten = tf.keras.layers.Flatten()    #17

        self.dense_ada8 = AdaptiveLayer(base_layer=basemodel.get_layer(index=18))
        self.bn6 = tf.keras.layers.BatchNormalization()

        self.dense_ada9 = AdaptiveLayer(base_layer=basemodel.get_layer(index=21))
        self.bn7 = tf.keras.layers.BatchNormalization()

        self.dense_ada10 = AdaptiveLayer(base_layer=basemodel.get_layer(index=24))



        #其它参数
        self.training = True
        #当处于训练状态时，默认inputs包含source，target domain同等数量的数据。

        #不能在类成员函数的参数里带self.******，任何self.的变量都不行
    def call(self,inputs):
        training = self.training
        #底层，复用部分
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

        #高层 计算MMD
        x = self.flatten(x)
        if training:
            x,mmk_1 = self.dense_ada8(x,training=training)
            x = self.bn6(x,training=training)
            x = tf.nn.relu(x)

            x,mmk_2 = self.dense_ada9(x,training=training)
            x = self.bn7(x,training=training)
            x = tf.nn.relu(x)

            logits,mmk_3 = self.dense_ada10(x,training=training)
            mmk = np.array([mmk_1,mmk_2,mmk_3])
            outputs = tf.nn.softmax(logits)

            return outputs,mmk
        else:
            x = self.dense_ada8(x,training=training)
            x = self.bn6(x,training=training)
            x = tf.nn.relu(x)

            x = self.dense_ada9(x,training=training)
            x = self.bn7(x,training=training)
            x = tf.nn.relu(x)

            logits = self.dense_ada10(x,training=training)
        
            outputs = tf.nn.softmax(logits)
        
            return outputs

#自适应层
class AdaptiveLayer(tf.keras.layers.Layer):
    def __init__(self,units=0,base_layer=None,activation=None,gamma=1):
        super().__init__()
        # 初始化代码
        self.units = units
        self.activation = activation
        self.gamma = gamma             #gammam用于选择核函数
        self.base_layer = base_layer
    
    def build(self, input_shape):
        # input_shape 是一个 TensorShape 类型对象，提供输入的形状
        # 在第一次使用该层的时候调用该部分代码，在这里创建变量可以使得变量的形状自适应输入的形状
        # 而不需要使用者额外指定变量形状。
        # 如果已经可以完全确定变量的形状，也可以在__init__部分创建变量
        self.w = self.add_weight(name='kernel',shape=[input_shape[-1],self.units],initializer=tf.zeros_initializer())  #因为是全连接层，所以权重shape 应取input最后一维
        self.b = self.add_weight(name="bias",shape=[self.units],initializer=tf.zeros_initializer())

    
    def call(self, inputs,training=False):
        # 模型调用的代码（处理输入并返回输出）
        #有多个输入时，tf会自动将第一个输入的shape赋给input_shape。保证多个输入的shape一致即可。

        if self.base_layer == None:
            outputs = tf.matmul(inputs,self.w) + self.b
            if self.activation != None :
                outputs = self.activation(outputs)
        else:
            outputs = self.base_layer(inputs)
        

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