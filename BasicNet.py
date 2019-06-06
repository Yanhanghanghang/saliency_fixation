import tensorflow as tf
import numpy as np

class BasicNet(object):
    weight_decay = 5*1e-6
    weight_init = 0.1 #weight init for bias
    leaky_alpha = 0.1
    is_training = False

    def __init__(self):
        self.pretrain_var_collection = []
        self.initial_var_collection = []
        self.trainable_var_collection = []
        #这个参数的作用是什么？
        self.var_rename = {}

    #自定义一个leaky_relu的激活函数，在x<0时，斜率为1，>0时斜率为0.1
    def leaky_relu(self, x, alpha, dtype = tf.float32):
        x = tf.cast(x, dtype=dtype)
        bool_mask = (x > 0)
        mask = tf.cast(bool_mask, dtype=dtype)
        return 1.0 * mask * x + alpha * (1 - mask) * x

    def get_bilinear(self, f_shape):
            width = f_shape[1]
            heigh = f_shape[0]
            f = width//2 + 1
            c = (2 * f - 1 - f % 2) / (2.0 * f)
            bilinear = np.zeros([f_shape[0], f_shape[1]])
            for x in range(width):
                for y in range(heigh):
                    value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                    bilinear[x, y] = value
            weights = np.zeros(f_shape)
            bilinear = bilinear / (np.sum(bilinear)*f_shape[2])
            for i in range(f_shape[2]):
                for j in range(f_shape[3]):
                    weights[:, :, i, j] = bilinear


            return weights
    
    def get_centermask(self,f_shape): # shape[batchsize, height, width, channals]
        width = f_shape[2]
        heigh = f_shape[1]
        midw = width//2
        midh = heigh//2
        distmatrix = np.zeros([heigh, width])
        for x in range(width):
            for y in range(heigh):
                value = np.sqrt((x - midw)**2+(y - midh)**2)
                distmatrix[x, y] = value
        distmatrix = distmatrix / np.max(distmatrix)
        distmatrix = 1 -  distmatrix
        distmatrix = distmatrix[np.newaxis,...,np.newaxis]
        # distmatrix = tf.expand_dims(distmatrix, 0)
        # distmatrix = tf.expand_dims(distmatrix, 3)
        # for a in range(f_shape[0]):
        #   for b in range(f_shape[3]):
        #     mask[a, :, :, b] = distmatrix
        return distmatrix

    def _activation_summary(self, x, name=None):
        if name is None:
            name = x.op.name
        #关注每一层激活函数的分布状况还有稀疏性（很重要）
        tf.summary.histogram(name + "/actications", x)
        tf.summary.scalar(name + "/sparsity", tf.nn.zero_fraction(x))

    def _variable_summaries(self, var):
        if not tf.get_variable_scope().reuse:
            name = var.op.name
            with tf.name_scope("summaries"):
                mean = tf.reduce_mean(var)
                tf.summary.scalar(name + "/mean", mean)
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var-mean)))
                tf.summary.scalar(name + '/stddev', stddev)
                l2norm = tf.sqrt(tf.reduce_sum(tf.square(var)))
                tf.summary.scalar(name + '/l2norm', l2norm)
                tf.summary.histogram(name, var)
    
    def _variable_on_cpu(self, name, shape, initializer, pretrain=False, trainable=True):
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)

        if not tf.get_variable_scope().reuse:
            if pretrain:
                self.pretrain_var_collection.append(var)
            else:
                self.initial_var_collection.append(var)
            if trainable:
                self.trainable_var_collection.append(var)

        return var
    
    def _variable_with_weight_decay(self, name, shape, wd, bilinear=False, pretrain=False, trainable=True):
        if bilinear:
            weights = self.get_bilinear(shape)
            initializer = tf.constant_initializer(value=weights, dtype=tf.float32)
        else:
            initializer = tf.contrib.layers.xavier_initializer()
        var = self._variable_on_cpu(name, shape, initializer, pretrain, trainable)

        if wd and not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_decay")
            weight_decay.set_shape([])
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)
        
        return var

    def conv_layer(self, scope_name, inputs, kernel_size, num_features, stride=1, linear=False, pretrain=False, trainable=True, batchnormalization=False):
        """ Convolutianal layer
        Args:
        input: 4-D
        tensor[batch_size, height, width, depth]
        scope_name: variable_scope
        kernel_size: [k_height, k_width]
        stride: int32

        Return:
        output: 4-D
        tensor[batch_size, height/stride, width/stride, num_features]
        """

        with tf.variable_scope(scope_name) as scope:
            input_features = inputs.get_shape()[3].value
            weights = self._variable_with_weight_decay('weights', shape=[kernel_size, kernel_size, input_features, num_features], wd=self.weight_decay, pretrain=pretrain, trainable=trainable)
            biases = self._variable_on_cpu('biases', [num_features], tf.constant_initializer(self.weight_init), pretrain, trainable)
            pad_size = kernel_size // 2
            pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
            inputs_pad = tf.pad(inputs, pad_mat)
            conv = tf.nn.conv2d(inputs_pad, weights, strides=[1, stride, stride, 1], padding='VALID')
            #?
#            self.testvar = biases
            conv_biased = tf.nn.bias_add(conv, biases, name='linearout')

            if batchnormalization:
                conv_biased = tf.layers.batch_normalization(conv_biased, training=self.is_training)
            if linear:
                return conv_biased
            conv_rect = self.leaky_relu(conv_biased, self.leaky_alpha)
            return conv_rect
    
    def transpose_conv_layer(self, scope_name, inputs, kernel_size, num_features, stride, linear=False, pretrain=False, trainable=False):
        with tf.variable_scope(scope_name) as scope:
            input_features = inputs.get_shape()[3].value
            weights = self._variable_with_weight_decay('weights', shape=[kernel_size, kernel_size, num_features, input_features], wd=self.weight_decay, pretrain=pretrain, bilinear=linear, trainable=trainable)
            biases = self._variable_on_cpu('biases', [num_features], tf.constant_initializer(self.weight_init), pretrain=pretrain, trainable=trainable)
            output_shape = tf.stack([tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], num_features])
            conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1, stride, stride, 1], padding='SAME')
            conv_biased = tf.nn.bias_add(conv, biases, name='linearout')
            
            if linear:
                return conv_biased

            conv_rect = self.leaky_relu(conv_biased, self.leaky_alpha)
            return conv_rect

    def max_pool(self, scope_name, inputs, kernel_size, stride):
        with tf.variable_scope(scope_name) as scope:
            pool = tf.nn.max_pool(inputs, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding='SAME', name='pooling')
        
        return pool
    
    def fc_layer(self, scope_name, inputs, hiddens, flat=False, linear=False, pretrain=False, trainable=True):
        with tf.variable_scope(scope_name) as scope:
            input_shape = inputs.get_shape().as_list()

            if flat:
                dim = input_shape[1]*input_shape[2]*input_shape[3]
                inputs_processed = tf.reshape(inputs, [-1, dim])
            else:
                dim = input_shape[1]
                inputs_processed = inputs
            
            weights = self._variable_with_weight_decay('weights', shape=[dim, hiddens], wd=self.weight_decay, pretrain=pretrain, trainable=trainable)
            biases = self._variable_on_cpu('biases',[hiddens],tf.constant_initializer(self.weight_init), pretrain, trainable)
            
            ip = tf.add(tf.matmul(inputs_processed, weights), biases, name='linearout')
            
            if linear:
                return ip
            fc_relu = self.leaky_relu(ip, self.leaky_alpha)
            return fc_relu

    def leaky_conv(self, net_in, n_filter, filter_size, strides, name, pretrain=True, trainable=True):
        return self.conv_layer(scope_name=name, inputs=net_in, kernel_size=filter_size, num_features=n_filter,
                                stride=strides, linear=False, pretrain=pretrain,
                                batchnormalization=False, trainable=trainable)

    def leaky_deconv(self, name, input_layer, n_filter, out_size):
        return self.transpose_conv_layer(scope_name=name, inputs=input_layer, kernel_size=4, num_features=n_filter,
                                            stride=2, linear=False, pretrain=True, trainable=True)

    def upsample(self, name, input_layer, out_size):
        return self.transpose_conv_layer(scope_name=name, inputs=input_layer, kernel_size=4, num_features=2,
                                            stride=2, linear=True, pretrain=True, trainable=True)

    def flow(self, name, input_layer, filter_size=3):
            return self.conv_layer(scope_name=name, inputs=input_layer, kernel_size=filter_size, num_features=2,
                                stride=1, linear=True, pretrain=True, batchnormalization=False, trainable=True)

    def conv_mask(self, net_in, mask):
        tempsize = net_in.get_shape().as_list()
        net_in_mask = tf.image.resize_images(mask, [tempsize[1], tempsize[2]])
        #print(net_in_mask.get_shape().as_list())
        return net_in * net_in_mask
            
    def kl_divergence(self, y_true, y_pred):
        # max_y_pred = K.repeat_elements(K.expand_dims(K.repeat_elements(K.expand_dims(K.max(K.max(y_pred, axis=1), axis=1)), shape_r_out, axis=1)), shape_c_out, axis=2)
#        max_y_pred = K.expand_dims(K.repeat_elements(
#            K.expand_dims(K.repeat_elements(K.expand_dims(K.max(y_pred, axis=[2, 3, 4])), y_pred.shape[2], axis=2)),
#            y_pred.shape[3], axis=3))        
        max_y_pred = y_pred
        for i in range(4, 1, -1):
            max_y_pred = tf.reduce_max(max_y_pred, axis=i)
        for i in range(2, 5, 1):
            max_y_pred = tf.expand_dims(max_y_pred, axis=i)       
        max_y_pred = tf.tile(max_y_pred, [1, 1, y_pred.shape[2], y_pred.shape[3], y_pred.shape[4]])
 
        y_pred /= max_y_pred

        max_y_true = y_true
        for i in range(4, 1, -1):
            max_y_true = tf.reduce_max(max_y_true, axis=i)
        for i in range(2, 5, 1):
            max_y_true = tf.expand_dims(max_y_true, axis=i)       
        max_y_true = tf.tile(max_y_true, [1, 1, y_true.shape[2], y_true.shape[3], y_true.shape[4]])

        y_bool = tf.cast(tf.greater(max_y_true, 0.1), 'float32')
        
        sum_y_pred = y_pred
        for i in range(4, 1, -1):
            sum_y_pred = tf.reduce_sum(sum_y_pred, axis=i)
        for i in range(2, 5, 1):
            sum_y_pred = tf.expand_dims(sum_y_pred, axis=i)       
        sum_y_pred = tf.tile(sum_y_pred, [1, 1, y_pred.shape[2], y_pred.shape[3], y_pred.shape[4]])

        sum_y_true = y_true
        for i in range(4, 1, -1):
            sum_y_true = tf.reduce_sum(sum_y_true, axis=i)
        for i in range(2, 5, 1):
            sum_y_true = tf.expand_dims(sum_y_true, axis=i)       
        sum_y_true = tf.tile(sum_y_true, [1, 1, y_true.shape[2], y_true.shape[3], y_true.shape[4]])


        y_true /= (sum_y_true + 1e-07)
        y_pred /= (sum_y_pred + 1e-07)
        return 10 * tf.reduce_sum(y_bool * y_true * tf.log((y_true / (y_pred + 1e-07) + 1e-07)))

    


            