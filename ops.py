import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


# 常数偏置
def bias(name, shape, bias_start=0.0, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.constant_initializer(
                              bias_start, dtype=dtype))
    return var


# 随机权重
def weight(name, shape, stddev=0.02, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.random_normal_initializer(
                              stddev=stddev, dtype=dtype))
    return var


# 全连接层
def fully_connected(value, output_shape, name='fully_connected', with_w=False):
    shape = value.get_shape().as_list()

    with tf.variable_scope(name):
        weights = weight('weights', [shape[1], output_shape], 0.02)
        biases = bias('biases', [output_shape], 0.0)

    if with_w:
        return tf.matmul(value, weights) + biases, weights, biases
    else:
        return tf.matmul(value, weights) + biases


# Leaky-ReLu 层
def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x, name=name)


# ReLu 层
def relu(value, name='relu'):
    with tf.variable_scope(name):
        return tf.nn.relu(value)


# 解卷积层
def deconv2d(value, output_shape, k_h=4, k_w=4, strides=[1, 2, 2, 1],
             name='deconv2d', with_w=False):
    with tf.variable_scope(name):
        weights = weight('weights',
                         [k_h, k_w, output_shape[-1], value.get_shape()[-1]])
        deconv = tf.nn.conv2d_transpose(value, weights,
                                        output_shape, strides=strides)
        biases = bias('biases', [output_shape[-1]])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, weights, biases
        else:
            return deconv


# 卷积层
def conv2d(value, biases, core,
            strides=[1, 1, 1, 1], name='conv2d'):
    with tf.variable_scope(name):
        conv = tf.nn.conv2d(value, core, strides=strides, padding='SAME')
        biases = tf.reshape(biases,[-1])
        conv = tf.nn.bias_add(conv, biases)
        return conv

def conv2d_noraml(value, output_dim, k_h=2, k_w=2,
            strides=[1, 1, 1, 1], name='conv2d'):
    with tf.variable_scope(name):
        weights = weight('weights',
                         [k_h, k_w, value.get_shape()[-1], output_dim])
        conv = tf.nn.conv2d(value, weights, strides=strides, padding='SAME')
        biases = bias('biases', [output_dim])
        conv = tf.nn.bias_add(conv, biases)

        return conv
# 把约束条件串联到 feature map
def conv_cond_concat(value, cond, name='concat'):
    # 把张量的维度形状转化成 Python 的 list
    value_shapes = value.get_shape().as_list()
    cond_shapes = cond.get_shape().as_list()

    # 在第三个维度上（feature map 维度上）把条件和输入串联起来，
    # 条件会被预先设为四维张量的形式，假设输入为 [64, 32, 32, 32] 维的张量，
    # 条件为 [64, 32, 32, 10] 维的张量，那么输出就是一个 [64, 32, 32, 42] 维张量
    with tf.variable_scope(name):
        return tf.concat(3, [value,
                             cond * tf.ones(value_shapes[0:3] + cond_shapes[3:])])


# Batch Normalization 层
def batch_norm_layer(value, is_train=True, name='batch_norm'):
    with tf.variable_scope(name) as scope:
        if is_train:
            return batch_norm(value, decay=0.9, epsilon=1e-5, scale=True,
                              is_training=is_train,
                              updates_collections=None, scope=scope)
        else:
            return batch_norm(value, decay=0.9, epsilon=1e-5, scale=True,
                              is_training=is_train, reuse=True,
                              updates_collections=None, scope=scope)

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial,name=name)


def conv2d_c(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

def max_pool(x, size):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding='VALID')