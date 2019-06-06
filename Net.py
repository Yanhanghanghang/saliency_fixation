import tensorflow as tf
import numpy as np
import BasicNet
from config import *

class Net(BasicNet.BasicNet):
    init_learning_rate = 1e-4
    eps = 1e-7
    gapnum = 5
    salmask_lb = 0.5
    cell_size = 7

    def __init__(self):
        super(Net, self).__init__()
        #process params
        self.global_step = tf.Variable(0, trainable=False)
        self.initial_var_collection.append(self.global_step)
        self.out = []
        self.predict = []
        self.loss = []
        self.loss_gt = []
        self.re = []
        self.loss_gt2 = []
        self.yolofeatures_collection = []
        self.flowfeatures_collection = []
        self.startflagcnn = True
    
    def YOLO_tiny_inference(self, images):
        cnnpretrain = True
        cnntrainable = False
        
        conv_1_1 = self.conv_layer('conv1', images, kernel_size=3, num_features=16, stride=1, pretrain=cnnpretrain, trainable=cnntrainable, batchnormalization=True)
        pool_1_2 = self.max_pool('pool2', conv_1_1, kernel_size=2, stride=2)
        conv_2_3 = self.conv_layer('conv3', pool_1_2, kernel_size=3, num_features=32, stride=1, pretrain=cnnpretrain, trainable=cnntrainable, batchnormalization=True)  
        pool_2_4 = self.max_pool('pool4', conv_2_3, kernel_size=2, stride=2)
        conv_3_5 = self.conv_layer('conv5', pool_2_4, kernel_size=3, num_features=64, stride=1, pretrain=cnnpretrain, trainable=cnntrainable, batchnormalization=True)  
        pool_3_6 = self.max_pool('pool6', conv_3_5, kernel_size=2, stride=2)
        conv_4_7 = self.conv_layer('conv7', pool_3_6, kernel_size=3, num_features=128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable, batchnormalization=True)  
        pool_4_8 = self.max_pool('pool8', conv_4_7, kernel_size=2, stride=2)
        conv_5_9 = self.conv_layer('conv9', pool_4_8, kernel_size=3, num_features=256, stride=1, pretrain=cnnpretrain, trainable=cnntrainable, batchnormalization=True)  
        pool_5_10 = self.max_pool('pool10', conv_5_9, kernel_size=2, stride=2)
        conv_6_11 = self.conv_layer('conv11', pool_5_10, kernel_size=3, num_features=512, stride=1, pretrain=cnnpretrain, trainable=cnntrainable, batchnormalization=True)  
        pool_6_12 = self.max_pool('pool12', conv_6_11, kernel_size=2, stride=2)
        conv_7_13 = self.conv_layer('conv13', pool_6_12, kernel_size=3, num_features=1024, stride=1, pretrain=cnnpretrain, trainable=cnntrainable, batchnormalization=False)  
        conv_8_14 = self.conv_layer('conv14', conv_7_13, kernel_size=3, num_features=1024, stride=1, pretrain=cnnpretrain, trainable=cnntrainable, batchnormalization=False)  
        conv_9_15 = self.conv_layer('conv15', conv_8_14, kernel_size=3, num_features=1024, stride=1, pretrain=cnnpretrain, trainable=cnntrainable, batchnormalization=False)  
          
        temp_conv = tf.transpose(conv_9_15, (0, 3, 1, 2))
        fc_1_16 = self.fc_layer('fc16', temp_conv, 256, flat=True, pretrain=cnnpretrain, trainable=cnntrainable)
        fc_2_17 = self.fc_layer('fc17', fc_1_16, 4096, flat=False, pretrain=cnnpretrain, trainable=cnntrainable)
        fc_3_18 = self.fc_layer('fc18', fc_2_17, 1470, flat=False, linear=True, pretrain=cnnpretrain, trainable=cnntrainable)

        highFeature = tf.reshape(fc_3_18, [fc_3_18.get_shape()[0], self.cell_size, self.cell_size, -1])

        conv_9_15_2 = self.conv_layer('conv_15_2', conv_9_15, kernel_size=1, num_features=128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
        conv_6_11_2 = self.conv_layer('conv_11_2', conv_6_11, kernel_size=1, num_features=128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
        conv_5_9_2 = self.conv_layer('con_9_2', conv_5_9, kernel_size=1, num_features=128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)

        tempsize = conv_5_9_2.get_shape().as_list()
        newconv_4_7 = tf.image.resize_images(conv_4_7, [tempsize[1], tempsize[2]])
        newconv_5_9 = tf.image.resize_images(conv_5_9_2, [tempsize[1], tempsize[2]])
        newconv_6_11 = tf.image.resize_images(conv_6_11_2, [tempsize[1], tempsize[2]])
        newconv_9_15 = tf.image.resize_images(conv_9_15_2, [tempsize[1], tempsize[2]])
        highFeature = tf.image.resize_images(highFeature, [tempsize[1], tempsize[2]])

        FeatureMap = tf.concat([newconv_4_7, newconv_5_9, newconv_6_11, newconv_9_15, highFeature], axis=3)
        centermask = tf.constant(self.get_centermask(FeatureMap.get_shape().as_list()), dtype=FeatureMap.dtype)
        FeatureMap = FeatureMap * centermask
        return FeatureMap
    
    def Coarse_salmap(self, Yolofeature):
        cnnpretrain = True
        cnntrainable = False
        conv_1 = self.conv_layer('Csconv_1', Yolofeature, kernel_size=3, num_features=512, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
        conv_2 = self.conv_layer('Csconv_2', conv_1, kernel_size=1, num_features=256, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
        conv_3 = self.conv_layer('Csconv_3', conv_2, kernel_size=3, num_features=128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
        conv_4 = self.conv_layer('Csconv_4', conv_3, kernel_size=1, num_features=128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)

        deconv_1 = self.transpose_conv_layer('Csdeconv_1', conv_4, kernel_size=4, num_features=16, stride=2, pretrain=cnnpretrain, trainable=cnntrainable)
        deconv_2 = self.transpose_conv_layer('Csdeconv_2', deconv_1, kernel_size=4, num_features=1, stride=2, linear=True, pretrain=cnnpretrain, trainable=cnntrainable)

        return deconv_2
    
    def Final_inference(self, cat1, cat2):
        cnnpretrain = True
        cnntrainable = False
        Myfeature = tf.concat([cat1, cat2], axis=3)
        Lastconv_1 = self.conv_layer('Lastconv_1', Myfeature, kernel_size=3, num_features=512, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
        Lastconv_2 = self.conv_layer('Lastconv_2', Lastconv_1, kernel_size=1, num_features=512, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
        Lastconv_3 = self.conv_layer('Lastconv_3', Lastconv_2, kernel_size=3, num_features=256, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
        Lastconv_4 = self.conv_layer('Lastconv_4', Lastconv_3, kernel_size=1, num_features=128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
        
        return Lastconv_4
    
    def flownet_with_conv(self, x1, x2, mask):
        cnnpretrain = True
        cnntrainable = False
        inputs = tf.concat([x1, x2], axis=3, name='FNinput')
        conv_1 = self.leaky_conv(net_in=inputs, n_filter=64, filter_size=7, strides=2, name='FNconv1', pretrain=cnnpretrain, trainable=cnntrainable)
        conv_1 = self.conv_mask(conv_1, mask)
        conv_2 = self.leaky_conv(net_in=conv_1, n_filter=128, filter_size=5, strides=2, name='FNconv2', pretrain=cnnpretrain, trainable=cnntrainable)
        conv_2 = self.conv_mask(conv_2, mask)
        conv_3 = self.leaky_conv(net_in=conv_2, n_filter=256, filter_size=5, strides=2, name='FNconv3', pretrain=cnnpretrain, trainable=cnntrainable)
        conv_3 = self.conv_mask(conv_3, mask)
        conv_3_1 = self.leaky_conv(net_in=conv_3, n_filter=256, filter_size=3, strides=1, name='FNconv3_1', pretrain=cnnpretrain, trainable=cnntrainable)
        conv_3_1 = self.conv_mask(conv_3_1, mask)
        conv_4 = self.leaky_conv(net_in=conv_3_1, n_filter=512, filter_size=3, strides=2, name='FNconv4', pretrain=cnnpretrain, trainable=cnntrainable)
        conv_4 = self.conv_mask(conv_4, mask)
        conv_4_1 = self.leaky_conv(net_in=conv_4, n_filter=512, filter_size=3, strides=1, name='FNconv4_1', pretrain=cnnpretrain, trainable=cnntrainable)
        conv_5 = self.leaky_conv(net_in=conv_4_1, n_filter=512, filter_size=3, strides=2, name='FNconv5', pretrain=cnnpretrain, trainable=cnntrainable)
        conv_5_1 = self.leaky_conv(net_in=conv_5, n_filter=512, filter_size=3, strides=1, name='FNconv5_1', pretrain=cnnpretrain, trainable=cnntrainable)
        conv_6 = self.leaky_conv(net_in=conv_5_1, n_filter=1024, filter_size=3, strides=2, name='FNconv6', pretrain=cnnpretrain, trainable=cnntrainable)
        conv_6_1 = self.leaky_conv(net_in=conv_6, n_filter=1024, filter_size=3, strides=1, name='FNconv6_1', pretrain=cnnpretrain, trainable=cnntrainable)
        out_cat_size = conv_4.get_shape().as_list()

        Downconv_6_1 = self.conv_layer('FNDownconv_6_1', conv_6_1, 3, 128, stride=1,pretrain=cnnpretrain, trainable=cnntrainable)
        Downconv_5_1 = self.conv_layer('FNDownconv_5_1', conv_5_1, 3, 128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
        Downconv_4_1 = self.conv_layer('FNDownconv_4_1', conv_4_1, 3, 128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
        Downconv_3_1 = self.conv_layer('FNDownconv_3_1', conv_3_1, 3, 128, stride=1, pretrain=cnnpretrain, trainable=cnntrainable)
        conv_6_1_cat = tf.image.resize_images(Downconv_6_1, [out_cat_size[1],out_cat_size[2]])
        conv_5_1_cat = tf.image.resize_images(Downconv_5_1, [out_cat_size[1],out_cat_size[2]])
        conv_4_1_cat = tf.image.resize_images(Downconv_4_1, [out_cat_size[1],out_cat_size[2]])
        conv_3_1_cat = tf.image.resize_images(Downconv_3_1, [out_cat_size[1],out_cat_size[2]])
        concat_out = tf.concat([conv_6_1_cat, conv_5_1_cat, conv_4_1_cat, conv_3_1_cat], axis=3, name='FNconcat_out')

        return concat_out

    #def inference(self, videoslides, mask_in, mask_h): #videoslides:[batch, framenum, h, w, num_features]
 #       with tf.variable_scope('inference') as scope:
    
    def _normlized(self, mat): # tensor [batch_size, image_height, image_width, channels] normalize each fea map
        mat_shape = mat.get_shape().as_list()
        tempsum = tf.reduce_sum(mat, axis=1)
        tempsum = tf.reduce_sum(tempsum, axis=1) + self.eps
        tempsum = tf.reshape(tempsum, [-1, 1, 1, mat_shape[3]])
        return mat / tempsum

    def _normlized_0to1(self, mat): # tensor [batch_size, image_height, image_width, channels] normalize each fea map
        mat_shape = mat.get_shape().as_list()
        tempmin = tf.reduce_min(mat, axis=1)
        tempmin= tf.reduce_min(tempmin, axis=1)
        tempmin = tf.reshape(tempmin, [-1, 1, 1, mat_shape[3]])
        tempmat = mat - tempmin
        tempmax = tf.reduce_max(tempmat, axis=1)
        tempmax = tf.reduce_max(tempmax, axis=1) + self.eps
        tempmax = tf.reshape(tempmax, [-1, 1, 1, mat_shape[3]])
        return tempmat / tempmax


    def _loss(self):
        weight_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope=None)
        loss_weight = tf.add_n(weight_loss)
        loss_kl = tf.get_collection('losses', scope=None)
        loss_kl = tf.add_n(loss_kl)/frame_num
        # self.out = self.predict
        tf.summary.scalar('loss_weight', loss_weight)
        tf.summary.scalar('loss_kl', loss_kl)
        self.loss_gt = loss_kl
        self.loss = loss_kl + loss_weight


    def inference(self, videoslides):
        with tf.variable_scope("inference"):
            shape = videoslides.get_shape().as_list()
            assert len(shape) == 5

            for indexframe in range(frame_num):
                frame = videoslides[:, indexframe, ...]
                frame_gap = videoslides[:, (indexframe + self.gapnum) % frame_num, ...]
                Yolo_features = self.YOLO_tiny_inference(frame)
                Presalmap = self.Coarse_salmap(Yolo_features)
                if self.startflagcnn == True:
                    self.yolofeatures_collection = self.pretrain_var_collection
                    self.pretrain_var_collection = []
                salmask = self._normlized_0to1(Presalmap)
                salmask = salmask*(1-self.salmask_lb)+self.salmask_lb
                Flow_features = self.flownet_with_conv(frame, frame_gap, salmask)
                CNNout = self.Final_inference(Yolo_features, Flow_features)
                
                output = self.Coarse_salmap(CNNout)
                norm_output = self._normlized_0to1(output)
                norm_output = tf.expand_dims(norm_output, 1)
                
                if indexframe == 0:
                    tempout = norm_output
                else:
                    tempout = tf.concat([tempout, norm_output], axis=1)
            self.out = tempout
    
    def kl_divergence(self, y_true, y_pred):
        max_y_pred = y_pred
        for i in range(4, 1, -1):
            max_y_pred = tf.reduce_max(max_y_pred, axis=i)
        for i in range(2, 5):
            max_y_pred = tf.expand_dims(max_y_pred, axis=i)
        max_y_pred = tf.tile(max_y_pred, [1, 1, y_pred.shape[2], y_pred.shape[3], y_pred.shape[4]])

        y_pred = y_pred / max_y_pred

        max_y_true = y_true
        for i in range(4, 1, -1):
            max_y_true = tf.reduce_max(max_y_true, axis=i)
        for i in range(2, 5):
            max_y_true = tf.expand_dims(max_y_true, axis=i)
        max_y_true = tf.tile(max_y_true, [1, 1, y_true.shape[2], y_true.shape[3], y_true.shape[4]])

        y_true = y_true / max_y_true

        y_bool = (max_y_true > 0.1)
        y_bool = tf.cast(y_bool, tf.float32)

        sum_y_true = y_true
        for i in range(4, 1, -1):
            sum_y_true = tf.reduce_sum(y_true, axis=i)
        for i in range(2, 5):
            sum_y_true = tf.expand_dims(sum_y_true, axis=i)
        sum_y_true = tf.tile(sum_y_true, [1, 1, y_true.shape[2], y_true.shape[3], y_true.shape[4]])

        sum_y_pred = y_pred
        for i in range(4, 1, -1):
            sum_y_pred = tf.reduce_sum(sum_y_pred, axis=i)
        for i in range(2, 5):
            sum_y_pred = tf.expand_dims(sum_y_pred, axis=i)
        sum_y_pred = tf.tile(sum_y_pred, [1, 1, y_pred.shape[2], y_pred.shape[3], y_pred.shape[4]])

        y_true = y_true / (sum_y_true + epsilon)
        y_pred = y_pred / (sum_y_pred + epsilon)
        return 10 * tf.reduce_sum(y_bool * y_true * tf.log((y_true / (y_pred + epsilon) + epsilon)))

    def correlation_coefficient(self, y_true, y_pred):

        max_y_pred = y_pred
        for i in range(4, 1, -1):
            max_y_pred = tf.reduce_max(max_y_pred, axis=i)
        for i in range(2, 5):
            max_y_pred = tf.expand_dims(max_y_pred, axis=i)
        max_y_pred = tf.tile(max_y_pred, [1, 1, y_pred.shape[2], y_pred.shape[3], y_pred.shape[4]])

        y_pred = y_pred / max_y_pred
        # max_y_true = K.expand_dims(K.repeat_elements(
        #     K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_true, axis=[2, 3, 4])), shape_r_out, axis=2)),
        #     shape_c_out, axis=3))
        max_y_true = y_true
        for i in range(4, 1, -1):
            max_y_true = tf.reduce_max(max_y_true, axis=i)
        y_bool = max_y_true > 0.1
        y_bool = tf.cast(y_bool, tf.float32)
        
        sum_y_pred = y_pred
        for i in range(4, 1, -1):
            sum_y_pred = tf.reduce_sum(sum_y_pred, axis=i)
        for i in range(2, 5):
            sum_y_pred = tf.expand_dims(sum_y_pred, axis=i)
        sum_y_pred = tf.tile(sum_y_pred, [1, 1, y_pred.shape[2], y_pred.shape[3], y_pred.shape[4]])

        sum_y_true = y_true
        for i in range(4, 1, -1):
            sum_y_true = tf.reduce_sum(sum_y_true, axis=i)
        for i in range(2, 5):
            sum_y_true = tf.expand_dims(sum_y_true, axis=i)
        sum_y_true = tf.tile(sum_y_true, [1, 1, y_true.shape[2], y_true.shape[3], y_true.shape[4]])

        y_true /= (sum_y_true + epsilon)
        y_pred /= (sum_y_pred + epsilon)

        N = y_pred._shape_as_list()[2] * y_pred._shape_as_list()[3]

        sum_prod = y_true * y_pred
        for i in range(4, 1, -1):
            sum_prod = tf.reduce_sum(sum_prod, axis=i)

        sum_x_square = y_true
        for i in range(4, 1, -1):
            sum_x_square = tf.reduce_sum(tf.square(sum_x_square), axis=i)
        sum_x_square = sum_x_square + epsilon 

        sum_y_square = y_pred
        for i in range(4, 1, -1):
            sum_y_square = tf.reduce_sum(tf.square(sum_y_square), axis=i)
        sum_y_square = sum_y_square + epsilon
        
        sum_x = y_true
        for i in range(4, 1, -1):
            sum_x = tf.reduce_sum(sum_x, axis=i)
        
        sum_y = y_pred
        for i in range(4, 1, -1):
            sum_y = tf.reduce_sum(sum_y, axis=i)

        num = sum_prod - ((sum_x * sum_y) / N)
        den = tf.sqrt((sum_x_square - tf.square(sum_x) / N) * (sum_y_square - tf.square(sum_y) / N))

        return tf.reduce_sum(y_bool*(-2 * num/den))#
    
    def nss(self, y_true, y_pred):
        max_y_pred = y_pred
        for i in range(4, 1, -1):
            max_y_pred = tf.reduce_max(max_y_pred, axis=i)
        for i in range(2, 5):
            max_y_pred = tf.expand_dims(max_y_pred, axis=i)
        max_y_pred = tf.tile(max_y_pred, [1, 1, y_pred.shape[2], y_pred.shape[3], y_pred.shape[4]])
        y_pred = y_pred / max_y_pred
        # y_pred_flatten = K.batch_flatten(y_pred)

        # max_y_true = K.expand_dims(K.repeat_elements(
        #     K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_true, axis=[2, 3, 4])), shape_r_out, axis=2)),
        #     shape_c_out, axis=3))
        max_y_true = y_true
        for i in range(4, 1, -1):
            max_y_true = tf.reduce_max(max_y_true, axis=i)
        y_bool = max_y_true > 0.1
        y_bool = tf.cast(y_bool, tf.float32)

        y_mean = y_pred
        for i in range(4, 1, -1):
            y_mean = tf.reduce_mean(y_mean, axis=i)
        for i in range(2, 5):
            y_mean = tf.expand_dims(y_mean, axis=i)
        y_mean = tf.tile(y_mean, [1, 1, y_pred.shape[2], y_pred.shape[3], y_pred.shape[4]])

        y_std = tf.square(y_pred - y_mean)
        for i in range(4, 1, -1):
            y_std = tf.reduce_mean(y_std, axis=i)
        y_std = tf.sqrt(y_std)
        for i in range(2, 5):
            y_std = tf.expand_dims(y_std, axis=i)
        
        y_pred = (y_pred - y_mean) / (y_std + epsilon)

        sum_y_true = y_true
        for i in range(4, 1, -1):
            sum_y_true = tf.reduce_sum(sum_y_true, axis=i)
        
        sum_prod = y_true * y_pred
        for i in range(4, 1, -1):
            sum_prod = tf.reduce_sum(sum_prod, axis=i)

        return -tf.reduce_sum(y_bool*(sum_prod) / (sum_y_true))