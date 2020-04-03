import tensorflow as tf
import random
import pathlib

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

