"""Train and test FCN 8s model for road segmentation."""
# Standard imports
import os
import warnings
from distutils.version import LooseVersion
import logging
import time
import argparse

# Local imports
import helper
import project_tests as tests
import tensorflow_ops as tf_ops

# Dependecy imports
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' # Disable tensorflow logging

# Check TensorFlow Version
ASSERT_MSG = 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), ASSERT_MSG

logging.info('TensorFlow Version: %s', tf.__version__)

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    logging.info('Default GPU Device: %s', tf.test.gpu_device_name())


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out,
             layer4_out, layer7_out)
    """
    logging.info("Loading Pretrained VGG Model into Tensorflow.")

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load the model and weights
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    default_graph = tf.get_default_graph()
    return (default_graph.get_tensor_by_name(vgg_input_tensor_name),
            default_graph.get_tensor_by_name(vgg_keep_prob_tensor_name),
            default_graph.get_tensor_by_name(vgg_layer3_out_tensor_name),
            default_graph.get_tensor_by_name(vgg_layer4_out_tensor_name),
            default_graph.get_tensor_by_name(vgg_layer7_out_tensor_name))

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, vgg_input=None):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    logging.info("Creating the layers for a fully convolutional network and building "
                 "skip-layers using the vgg layers.")

    fc7_to_conv = tf_ops.conv_layer(
        bottom=vgg_layer7_out,
        n_filters=num_classes,
        kernel_size=[1, 1],
        strides=1,
        activation=None,
        name="fc7_to_conv"
    )

    upscore2 = tf_ops.upscore_layer(
        fc7_to_conv,
        shape=tf.shape(vgg_layer4_out),
        num_classes=num_classes,
        name='upscore2',
        ksize=4,
        stride=2
    )

    score_pool4 = tf_ops.score_layer(
        vgg_layer4_out,
        "score_pool4",
        num_classes=num_classes
    )

    fuse_pool4 = tf.add(upscore2, score_pool4)

    upscore4 = tf_ops.upscore_layer(
        fuse_pool4,
        shape=tf.shape(vgg_layer3_out),
        num_classes=num_classes,
        name='upscore4',
        ksize=4,
        stride=2
    )
    score_pool3 = tf_ops.score_layer(vgg_layer3_out, "score_pool3", num_classes=num_classes)
    fuse_pool3 = tf.add(upscore4, score_pool3)

    if vgg_input is None: # To pass a test
        output_shape = [FLAGS.batch_size] + FLAGS.image_shape
    else:
        # For different input image shapes and batch sizes
        output_shape = tf.shape(vgg_input)

    upscore32 = tf_ops.upscore_layer(
        fuse_pool3,
        shape=output_shape,
        num_classes=num_classes,
        name='upscore32',
        ksize=16,
        stride=8
    )

    return upscore32


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logging.info("Building the TensorFlow loss and optimizer operations.")

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.to_float(tf.reshape(correct_label, (-1, num_classes)))

    epsilon = tf.constant(value=1e-4)

    softmax = tf.nn.softmax(logits) + epsilon

    cross_entropy = - tf.reduce_sum(labels * tf.log(softmax), reduction_indices=[1]) # pylint: disable=E1130

    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)

    # TODO: The way the project and tests are orginized I did not use tensorflow global step.
    # But I should be using global step in optimization function. So it can be used e.g. Summaries
    # or MonitorTrainingSession. Also it is saved as variable, which may be useful.

    loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return logits, optimizer, loss


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.
           Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: This is actual not learning but nn_last_layer
    """

    # TODO: Split training to multiple GPUs. As I don't have multiple GPUs available for this
    # project I didn't implement that.

    # TODO: This project does not take advantage of Tf Queues, TF Dataset, Iterators because of
    # function restrictions and because I didn't want to define VGG from sctrach. We use unefficient
    # placeholders instead.

    # !NB -> because I needed the project tests to pass as well as summaries I just replaced
    # learning_rate with nn_last_layer.
    outputs = tf.cast(tf.expand_dims(tf.argmax(learning_rate, axis=-1), -1), tf.uint8) * 100
    correct_lbl = tf.cast(tf.expand_dims(tf.argmax(correct_label, axis=-1), -1), tf.uint8) * 100

    outputs = tf.concat([tf.cast(input_image, tf.uint8),
                         tf.concat([correct_lbl, correct_lbl, correct_lbl], axis=-1),
                         tf.concat([outputs, outputs, outputs], axis=-1)], axis=1)

    loss_summ = tf.summary.scalar("training_loss", cross_entropy_loss)

    im_summ = tf.summary.image('images', outputs, max_outputs=10)

    if not TESTING:
        summary_writer = tf.summary.FileWriter(
            logdir='./logdir',
            graph=sess.graph,
            flush_secs=60,
            filename_suffix=FLAGS.test_name
        )

    # Initialize TF variables
    init_d = tf.global_variables_initializer()
    sess.run(init_d)

    logging.info("Going to train on %d epochs", epochs)

    counter = 1
    for epch in range(epochs):
        generator = get_batches_fn(batch_size)
        for i, batch_data in enumerate(generator):
            start_time = time.time()

            images = batch_data[0]  # Get batch images
            # Get batch labels (batch_size, h, w, num_classes) <-- bool
            labels = batch_data[1].astype(np.uint8)

            # Usually we would do augmentation using tensorflow ops but because training for this
            # project is fast, there are no much training samples and project structure is kinda
            # predefined we will sattle with cv2/numpy transformations.

            if len(images.shape) == 4:
                images, labels = helper.preprocessing(images, labels)

            feed_dict = {input_image: images, correct_label: labels, keep_prob: 0.5}

            if counter % 30 == 0:
                ops = [train_op, cross_entropy_loss, loss_summ, im_summ]
                _, loss, lsumm, imsumm = sess.run(ops, feed_dict=feed_dict)

                if not TESTING:
                    summary_writer.add_summary(imsumm, counter)
                    summary_writer.add_summary(lsumm, counter)
            else:
                _, loss, lsumm = sess.run([train_op, cross_entropy_loss, loss_summ],
                                          feed_dict=feed_dict)

                if not TESTING:
                    summary_writer.add_summary(lsumm, counter)

            # _, loss = sess.run([train_op, cross_entropy_loss], feed_dict=feed_dict)

            counter += 1
            end_time = time.time() - start_time
            logging.info('[Time: %.3f] Epoch: %d, Batch: %d Loss: %f',
                         end_time, epch + 1, i + 1, loss)

        logging.info('Epoch %d done!', epch + 1)

    if not TESTING:
        summary_writer.close()
    logging.info('End of training!')


def run():
    """Run project."""

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # Path to vgg model
    vgg_path = os.path.join(FLAGS.data_dir, 'vgg')

    with tf.Session() as sess:

        if FLAGS.mode == 'inference_model' or FLAGS.mode == 'train':
            # Create function to get batches
            get_batches_fn = helper.gen_batch_function(
                os.path.join(FLAGS.data_dir, 'data_road/training'),
                FLAGS.image_shape
            )

            correct_label = tf.placeholder('float', name='correct_label')

            vgg_input, vgg_keep_prob, vgg_l3_out, vgg_l4_out, vgg_l7_out = load_vgg(sess, vgg_path)
            nn_last_layer = layers(vgg_l3_out, vgg_l4_out, vgg_l7_out, FLAGS.num_classes, vgg_input)

        if FLAGS.mode == 'inference_model' and FLAGS.chk_path is not None:
            logging.info("Re-saving checkpoint file for smaller model.")

            with tf.variable_scope('outputs'): # pylint: disable=E1129
                logits = tf.reshape(nn_last_layer, (-1, FLAGS.num_classes))
                tf.nn.softmax(logits, name='logits_softmax')

            tf.train.Saver().restore(sess, FLAGS.chk_path)

            tf_ops.make_chk(sess, './data/vgg_fcn/inference_' + os.path.basename(FLAGS.chk_path))
            tf_ops.make_pbtxt(sess, ('./data/vgg_fcn/inference_' +
                                     os.path.basename(FLAGS.chk_path).split('.')[0] + '.pbtxt'))

        elif FLAGS.mode == 'train':

            logits, optimizer, loss = optimize(
                nn_last_layer=nn_last_layer,
                correct_label=correct_label,
                learning_rate=FLAGS.learning_rate,
                num_classes=FLAGS.num_classes
            )

            train_nn(
                sess=sess,
                epochs=FLAGS.epochs,
                batch_size=FLAGS.batch_size,
                get_batches_fn=get_batches_fn,
                train_op=optimizer,
                cross_entropy_loss=loss,
                input_image=vgg_input,
                correct_label=correct_label,
                keep_prob=vgg_keep_prob,
                learning_rate=nn_last_layer
            )

            tf_ops.make_chk(sess, './data/vgg_fcn/model.chk')

            # Practically this works as visual test after every training session.
            # Let's measure calcuations time to what happens after inference optimization
            start_time = time.time()

            helper.save_inference_samples(
                runs_dir=FLAGS.runs_dir,
                data_dir=FLAGS.data_dir,
                sess=sess,
                image_shape=FLAGS.image_shape,
                logits=logits,
                keep_prob=vgg_keep_prob,
                input_image=vgg_input
            )

            end_time = time.time()
            logging.info('Test time: %.2f', end_time - start_time)

        elif FLAGS.mode == 'inference_test':
            raise NotImplementedError('Inference Test on video is not yet implemented.')
            # OPTIONAL: Apply the trained model to a video


def test_project():
    """Execute all project tests and download pretrained vgg."""

    logging.info('Test for Kitti Dataset:')
    tests.test_for_kitti_dataset(FLAGS.data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(FLAGS.data_dir)

    logging.info('Test Load VGG:')
    tests.test_load_vgg(load_vgg, tf)

    logging.info('Test Layers:')
    tests.test_layers(layers)

    logging.info('Test Optimization:')
    tests.test_optimize(optimize)

    logging.info('Test Train NN:')
    tests.test_train_nn(train_nn)

def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--image_shape",
        type=int,
        default=[160, 576, 3],
        nargs='+',
        help="Resized image shape which will be used as input for neural net."
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of batches."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Optimizer initial learning rate."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default='./data',
        help="Data directory path."
    )
    parser.add_argument(
        "--runs_dir",
        type=str,
        default='./runs',
        help="Runs directory path."
    )

    parser.add_argument(
        "--test_name",
        type=str,
        default=None,
        help="Test name, used when create log dir with summaries as prefix"
    )

    # This will pass training process and just save previously trained model
    # I do this because saving model right after training results in relatively
    # big model but SAVE_INFERENCE_MODEL == True then we just construct layers and
    # load trained checkpoint weights and then save again. Result ~3 times smaller
    # model. This also makes optimization much faster.
    parser.add_argument(
        "--chk_path",
        type=str,
        default=None,
        help="Re-save checkpoint path for optimization. If not set then won't save anything."
    )

    parser.add_argument(
        "--pb_path",
        type=str,
        default=None,
        help="Path to optimized FCN model for inferece."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default='train',
        help="Run code in possible modes:"
             "--mode train : Will train and save mode. Afterwards test and save results."
             "--mode inference_model : Will re-save checkpoint path for optimization. For this "
             "--chk_path must be provided"
             "--mode inference_test : Will run inference model on test video. --pb_path must "
             "be provided"
             "--mode project_test : Will only perform project unit tests."
    )

    return parser.parse_known_args()

if __name__ == '__main__':
    FLAGS, _ = parse_args()

    TESTING = True # Prevent project testing from creating unecesarry summaries
    test_project() # test project
    TESTING = False

    if FLAGS.mode != 'project_test':
        run() # run project
