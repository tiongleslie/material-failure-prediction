from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation

# Hide all the warning messages from TensorFlow
deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


class DML(object):
    def __init__(self):
        print('DML pretrain model loading...\n\n')

    @staticmethod
    def DML_model(path, x_1=None, x_2=None):
        """
        This function loads the weights from pretrained model.

        :param path: path of the pretrain model
        :param x_1: the input of PD image
        :param x_2: the input of PD-extracted metric
        :return: prediction result
        """
        graph = tf.get_default_graph()
        with tf.Session(config=config) as sess:
            saver = tf.train.import_meta_graph(path + '/DML.meta')
            saver.restore(sess, path + '/DML')

            x1 = graph.get_tensor_by_name("input_img:0")
            x2 = graph.get_tensor_by_name("input_x:0")
            drop_prob = graph.get_tensor_by_name("keep_prob:0")

            pred_ = graph.get_tensor_by_name("ann_net/output/add:0")

            input_feed = {x1: x_1, x2: x_2, drop_prob: 1.0}
            result_ = sess.run(pred_/2, feed_dict=input_feed)

        return result_.reshape(-1)[0] * 100
