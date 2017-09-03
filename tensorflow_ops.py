"""Implement simple functions to be used for tensorflow for make code more readable in main.py"""

# Standard imports
import math
import logging
import os

# Dependecy imports
import tensorflow as tf
import numpy as np

W_DECAY = 5e-4

def get_deconv_filter(f_shape):
    """Create deconvolution tf variable."""

    width = f_shape[0]
    heigh = f_shape[0]

    features = math.ceil(width / 2.0)
    _conv = (2 * features - 1 - features % 2) / (2.0 * features)
    bilinear = np.zeros([f_shape[0], f_shape[1]])

    for i_w in range(width):
        for i_h in range(heigh):
            value = (1 - abs(i_w / features - _conv)) * (1 - abs(i_h / features - _conv))
            bilinear[i_w, i_h] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    var = tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)

    return var

def variable_with_weight_decay(shape, stddev, wd_mul, decoder=False):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal
    distribution.
    A weight decay is added only if one is specified.
    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd_mul: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.
    Returns:
        Variable Tensor
    """

    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable('weights', shape=shape, initializer=initializer)

    if wd_mul and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(
            tf.nn.l2_loss(var), wd_mul, name='weight_loss')
        if not decoder:
            tf.add_to_collection('losses', weight_decay)
        else:
            tf.add_to_collection('dec_losses', weight_decay)

    return var

def bias_variable(shape, constant=0.0):
    """Initialize bias variable with constant value."""

    initializer = tf.constant_initializer(constant)
    var = tf.get_variable(name='biases', shape=shape, initializer=initializer)

    return var

def conv_layer(bottom, n_filters, kernel_size, strides, activation, name):
    """TF Convolution layer."""

    with tf.variable_scope(name) as scope: # pylint: disable=E1129

        output = tf.layers.conv2d(
            inputs=bottom,
            filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
            padding='SAME',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
        )

        logging.debug("Conv Layer: %s Filters: %s", scope, n_filters)

        return output

def score_layer(bottom, name, num_classes):
    """Score layer."""

    with tf.variable_scope(name) as scope: # pylint: disable=E1129
        # get number of input channels
        in_features = bottom.get_shape()[3].value
        shape = [1, 1, in_features, num_classes]

        # He initialization Sheme
        if name == "score_fr":
            num_input = in_features
            stddev = (2 / num_input)**0.5
        elif name == "score_pool4":
            stddev = 0.001
        elif name == "score_pool3":
            stddev = 0.0001
        # Apply convolution

        logging.debug("Score Layer: %s, Fan-in: %d, Shape: %s nClass: %d",
                      scope, in_features, shape, num_classes)

        weights = variable_with_weight_decay(shape, stddev, W_DECAY, decoder=True)
        conv = tf.nn.conv2d(bottom, weights, [1, 1, 1, 1], padding='SAME')
        # Apply bias
        conv_biases = bias_variable([num_classes], constant=0.0)
        bias = tf.nn.bias_add(conv, conv_biases)

        return bias

def conv_transpose_layer(bottom, n_filters, kernel_size, strides, activation, name):
    """TF Convolution transpose layer."""

    with tf.variable_scope(name) as scope: # pylint: disable=E1129

        output = tf.layers.conv2d_transpose(
            inputs=bottom,
            filters=n_filters,
            kernel_size=kernel_size,
            strides=strides,
            activation=activation,
            padding='SAME',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3)
        )

        logging.debug("Conv Layer: %s Filters: %s", scope, n_filters)

        return output

def upscore_layer(bottom, shape, num_classes, name, ksize=4, stride=2):
    """Upscore layer."""
    strides = [1, stride, stride, 1]

    with tf.variable_scope(name): # pylint: disable=E1129
        in_features = bottom.get_shape()[3].value
        new_shape = [shape[0], shape[1], shape[2], num_classes]

        output_shape = tf.stack(new_shape)

        logging.debug("Upscore Layer: %s, Fan-in: %d, Shape: %s nClass: %d Output shape: %s",
                      name, in_features, new_shape, num_classes, output_shape)

        f_shape = [ksize, ksize, num_classes, in_features]

        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')

    return deconv

def make_pbtxt(sess, save_path):
    """Load net, session and weights and create protobuf text file."""

    tf.train.write_graph(
        sess.graph_def,
        os.path.dirname(save_path),
        os.path.basename(save_path)
    )

    logging.info("pbtxt saved: %s", save_path)

def make_chk(sess, save_path):
    """Load net, session and weights and create checkpoint file."""

    saver = tf.train.Saver()
    saver.save(
        sess,
        save_path,
        write_meta_graph=False
    )

    logging.info("chk saved: %s", save_path)
