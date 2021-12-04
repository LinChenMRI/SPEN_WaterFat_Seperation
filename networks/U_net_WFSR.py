from networks.ops import *
import tensorflow as tf
import datetime
import numpy as np
# parameters
flags = tf.flags
flags.DEFINE_integer('MAX_INTER', 150000, 'The number of training steps')
flags.DEFINE_integer('MAX_TO_KEEP', 1, 'The max number of model to save')
flags.DEFINE_integer('BATCH_SIZE', 8, 'The size of batch images [8]')
flags.DEFINE_float('BETA', 1e-6, 'TV Optimizer [8e-2]')
flags.DEFINE_integer(
    'STEP', None,
    'Which checkpoint should be load, None for final step of checkpoint [None]'
)
flags.DEFINE_float('LR', 1e-4, 'Learning rate of for Optimizer [1e-4]')
flags.DEFINE_integer('NUM_GPUS', 1, 'The number of GPU to use [1]')
flags.DEFINE_boolean('IS_TRAIN', True, 'True for train, else test. [True]')
flags.DEFINE_integer(
    'FILTER_DIM', 64,
    'The number of feature maps in all layers. [64]'
)
flags.DEFINE_boolean(
    'LOAD_MODEL', True,
    'True for load checkpoint and continue training. [True]'
)
# flags.DEFINE_string(
#     'MODEL_DIR', 'UNet_trained_for_water',
#     'If LOAD_MODEL, provide the MODEL_DIR. [./model/baseline/]'
# )
flags.DEFINE_string(
    'DATA_DIR', 'reconstruction_result',
    'the data set directihtopon'
)
FLAGS = flags.FLAGS

#deep 5
def model(images, reuse = False, name='UNet'):

    with tf.variable_scope(name, reuse=reuse):
        L1_1 = conv_relu(images, [3, 3, FLAGS.INPUT_C, FLAGS.FILTER_DIM], 1)
        L1_2 = conv_relu(L1_1, [3, 3, 64, FLAGS.FILTER_DIM], 1)
        L2_1 = tf.nn.max_pool(L1_2, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')  ##

        L2_2 = conv_relu(L2_1, [3, 3, FLAGS.FILTER_DIM, FLAGS.FILTER_DIM*2], 1)
        L2_3 = conv_relu(L2_2, [3, 3, FLAGS.FILTER_DIM*2, FLAGS.FILTER_DIM*2], 1)
        L3_1 = tf.nn.max_pool(L2_3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')    ##

        L3_2 = conv_relu(L3_1, [3, 3, FLAGS.FILTER_DIM*2, FLAGS.FILTER_DIM*4], 1)
        L3_3 = conv_relu(L3_2, [3, 3, FLAGS.FILTER_DIM*4, FLAGS.FILTER_DIM*4], 1)
        L4_1 = tf.nn.max_pool(L3_3, [1, 2, 2, 1], [1, 2, 2, 1], padding = 'SAME')    ##

        L4_2 = conv_relu(L4_1, [3, 3, FLAGS.FILTER_DIM*4, FLAGS.FILTER_DIM*8], 1)
        L4_3 = conv_relu(L4_2, [3, 3, FLAGS.FILTER_DIM*8, FLAGS.FILTER_DIM*8], 1)
        L5_1 = tf.nn.max_pool(L4_3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')  ##

        L5_2 = conv_relu(L5_1, [3, 3, FLAGS.FILTER_DIM * 8, FLAGS.FILTER_DIM * 16], 1)
        L5_3 = conv_relu(L5_2, [3, 3, FLAGS.FILTER_DIM * 16, FLAGS.FILTER_DIM * 16], 1)

        L4_U1 = deconv2(L5_3, FLAGS.FILTER_DIM * 16, FLAGS.FILTER_DIM * 8, 2, 2)
        L4_U1 = tf.concat((L4_3, L4_U1), 3)
        L4_U2 = conv_relu(L4_U1, [3, 3, FLAGS.FILTER_DIM * 16, FLAGS.FILTER_DIM * 8], 1)
        L4_U3 = conv_relu(L4_U2, [3, 3, FLAGS.FILTER_DIM * 8, FLAGS.FILTER_DIM * 8], 1)

        L3_U1 = deconv2(L4_U3, FLAGS.FILTER_DIM * 8, FLAGS.FILTER_DIM * 4, 2, 2)
        L3_U1 = tf.concat((L3_3, L3_U1), 3)
        L3_U2 = conv_relu(L3_U1, [3, 3, FLAGS.FILTER_DIM * 8, FLAGS.FILTER_DIM * 4], 1)
        L3_U3 = conv_relu(L3_U2, [3, 3, FLAGS.FILTER_DIM * 4, FLAGS.FILTER_DIM * 4], 1)

        L2_U1 = deconv2(L3_U3, FLAGS.FILTER_DIM * 4, FLAGS.FILTER_DIM * 2, 2, 2)
        L2_U1 = tf.concat((L2_3, L2_U1), 3)
        L2_U2 = conv_relu(L2_U1, [3, 3, FLAGS.FILTER_DIM * 4, FLAGS.FILTER_DIM * 2], 1)
        L2_U3 = conv_relu(L2_U2, [3, 3, FLAGS.FILTER_DIM * 2, FLAGS.FILTER_DIM * 2], 1)

        L1_U1 = deconv2(L2_U3, FLAGS.FILTER_DIM * 2, FLAGS.FILTER_DIM * 1, 2, 2)
        L1_U1 = tf.concat((L1_2, L1_U1), 3)
        L1_U2 = conv_relu(L1_U1, [3, 3, FLAGS.FILTER_DIM * 2, FLAGS.FILTER_DIM * 1], 1)
        L1_U3 = conv_relu(L1_U2, [3, 3, FLAGS.FILTER_DIM * 1, FLAGS.FILTER_DIM * 1], 1)

        out = conv_relu(L1_U3, [3, 3, FLAGS.FILTER_DIM, FLAGS.OUTPUT_C], 1)



    # variables = tf.contrib.framework.get_variables(name)

    return out

def losses(output, labels, name = 'losses'):
    with tf.name_scope(name):
        loss_water = tf.reduce_mean(tf.square(output[:, :, :, :] - labels[:, :, :, :]))  # L2 regularization
        loss = loss_water
        return loss, loss_water