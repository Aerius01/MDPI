import os
import numpy as np
from typing import Tuple, Dict
from .config import ClassificationConfig

# Suppress TensorFlow logging messages - MUST be set before importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=all, 1=no INFO, 2=no INFO/WARN, 3=no INFO/WARN/ERROR

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class CNNArchitecture:
    """Handles CNN model construction with TensorFlow 1.x syntax."""
    
    def __init__(self, config: ClassificationConfig):
        self.config = config
        self.nr_categories = len(config.categories)
        
    def build_model(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Constructs the CNN architecture and returns placeholders."""
        # Define placeholders
        x_input = tf.compat.v1.placeholder(tf.float32, shape=[None, self.config.input_size, self.config.input_size])
        keep_prob = tf.compat.v1.placeholder(tf.float32)
        
        # Build network
        y_pred = self._build_network(x_input, keep_prob)
        
        return x_input, keep_prob, y_pred
    
    def _build_network(self, x_input: tf.Tensor, keep_prob: tf.Tensor) -> tf.Tensor:
        """Builds the complete neural network architecture."""
        # LRN parameters
        lrn_params = {'depth_radius': 2, 'alpha': 2e-05, 'beta': 0.75, 'bias': 1.0}
        
        # Reshape input
        x_image = tf.reshape(x_input, [-1, self.config.input_size, self.config.input_size, self.config.input_depth])
        
        # Convolutional blocks
        conv1_out = self._conv_block_1(x_image, lrn_params)
        conv2_out = self._conv_block_2(conv1_out, lrn_params)
        conv3_out = self._conv_block_3(conv2_out)
        
        # Fully connected layers
        y_pred = self._fc_layers(conv3_out, keep_prob)
        
        return y_pred
    
    def _conv_block_1(self, x_image: tf.Tensor, lrn_params: Dict) -> tf.Tensor:
        """First convolutional block: [batch, 50, 50, 1] -> [batch, 12, 12, 32]"""
        conv1_filters = tf.Variable(tf.random.truncated_normal([5, 5, 1, 32], stddev=0.05), name="conv1W")
        conv1_bias = tf.Variable(tf.constant(0.1, shape=[32]), name="conv1b")
        conv1_in = tf.compat.v1.nn.conv2d(x_image, conv1_filters, [1, 2, 2, 1], padding='SAME')
        conv1_out = tf.reshape(tf.nn.bias_add(conv1_in, conv1_bias), [-1] + conv1_in.get_shape().as_list()[1:])
        conv1_relu = tf.compat.v1.nn.relu(conv1_out)
        conv1_norm = tf.compat.v1.nn.local_response_normalization(conv1_relu, **lrn_params)
        return tf.compat.v1.nn.max_pool2d(conv1_norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    def _conv_block_2(self, conv1_out: tf.Tensor, lrn_params: Dict) -> tf.Tensor:
        """Second convolutional block: [batch, 12, 12, 32] -> [batch, 5, 5, 64]"""
        conv2_filters = tf.Variable(tf.random.truncated_normal([5, 5, 32, 64], stddev=0.05), name="conv2W")
        conv2_bias = tf.Variable(tf.constant(0.1, shape=[64]), name="conv2b")
        conv2_in = tf.compat.v1.nn.conv2d(conv1_out, conv2_filters, [1, 1, 1, 1], padding='SAME')
        conv2_out = tf.reshape(tf.nn.bias_add(conv2_in, conv2_bias), [-1] + conv2_in.get_shape().as_list()[1:])
        conv2_relu = tf.compat.v1.nn.relu(conv2_out)
        conv2_norm = tf.compat.v1.nn.local_response_normalization(conv2_relu, **lrn_params)
        return tf.compat.v1.nn.max_pool2d(conv2_norm, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    def _conv_block_3(self, conv2_out: tf.Tensor) -> tf.Tensor:
        """Third convolutional block: [batch, 5, 5, 64] -> [batch, 2, 2, 96]"""
        # Conv3
        conv3_filters = tf.Variable(tf.random.truncated_normal([3, 3, 64, 96], stddev=0.05), name="conv3W")
        conv3_bias = tf.Variable(tf.constant(0.1, shape=[96]), name="conv3b")
        conv3_in = tf.compat.v1.nn.conv2d(conv2_out, conv3_filters, [1, 1, 1, 1], padding='SAME')
        conv3_out = tf.reshape(tf.nn.bias_add(conv3_in, conv3_bias), [-1] + conv3_in.get_shape().as_list()[1:])
        conv3_relu = tf.compat.v1.nn.relu(conv3_out)
        
        # Conv4
        conv4_filters = tf.Variable(tf.random.truncated_normal([3, 3, 96, 96], stddev=0.05), name="conv4W")
        conv4_bias = tf.Variable(tf.constant(0.1, shape=[96]), name="conv4b")
        conv4_in = tf.compat.v1.nn.conv2d(conv3_relu, conv4_filters, [1, 1, 1, 1], padding='SAME')
        conv4_out = tf.reshape(tf.nn.bias_add(conv4_in, conv4_bias), [-1] + conv4_in.get_shape().as_list()[1:])
        conv4_relu = tf.compat.v1.nn.relu(conv4_out)
        
        # Conv5
        conv5_filters = tf.Variable(tf.random.truncated_normal([3, 3, 96, 96], stddev=0.05), name="conv5W")
        conv5_bias = tf.Variable(tf.constant(0.1, shape=[96]), name="conv5b")
        conv5_in = tf.compat.v1.nn.conv2d(conv4_relu, conv5_filters, [1, 1, 1, 1], padding='SAME')
        conv5_out = tf.reshape(tf.nn.bias_add(conv5_in, conv5_bias), [-1] + conv5_in.get_shape().as_list()[1:])
        conv5_relu = tf.compat.v1.nn.relu(conv5_out)
        
        return tf.compat.v1.nn.max_pool2d(conv5_relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    
    def _fc_layers(self, conv_out: tf.Tensor, keep_prob: tf.Tensor) -> tf.Tensor:
        """Fully connected layers: [batch, 2, 2, 96] -> [batch, 4]"""
        # FC6
        fc6_nodes = tf.Variable(tf.random.truncated_normal([384, 76], stddev=0.1), name="fc6")
        fc6_bias = tf.Variable(tf.constant(0.1, shape=[76]), name="fc6")
        fc6_relu = tf.compat.v1.nn.relu_layer(
            tf.reshape(conv_out, [-1, int(np.prod(conv_out.get_shape()[1:]))]), 
            fc6_nodes, fc6_bias
        )
        fc6_drop = tf.compat.v1.nn.dropout(fc6_relu, rate=1 - keep_prob)
        
        # FC7
        fc7_nodes = tf.Variable(tf.random.truncated_normal([76, 76], stddev=0.1), name="fc7")
        fc7_bias = tf.Variable(tf.constant(0.1, shape=[76]), name="fc7")
        fc7_relu = tf.compat.v1.nn.relu_layer(fc6_drop, fc7_nodes, fc7_bias)
        fc7_drop = tf.compat.v1.nn.dropout(fc7_relu, rate=1 - keep_prob)
        
        # FC8
        fc8_nodes = tf.Variable(tf.random.truncated_normal([76, self.nr_categories], stddev=0.1), name="fc8")
        fc8_bias = tf.Variable(tf.constant(0.1, shape=[self.nr_categories]), name="fc8")
        return tf.compat.v1.nn.xw_plus_b(fc7_drop, fc8_nodes, fc8_bias) 