import tensorflow as tf

import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import cv2
from numpy import asarray
from numpy import save

import datetime

imagefile = "../MNIST_data_unziped/t10k-images.idx3-ubyte"
imagearray = idx2numpy.convert_from_file(imagefile)

labelfile = "../MNIST_data_unziped/t10k-labels.idx1-ubyte"
labelarray = idx2numpy.convert_from_file(labelfile)

y_Predicted = []


def img_normal(raw):
    # normalize pixels to 0 and 1.
    # digit is 1, background is 0
    norm = np.array(raw, np.float32)
    norm /= 255.0
    return norm


def detect_all_img():
    sess = tf.Session()

    # load meta
    saver = tf.train.import_meta_graph('../model/AlexNet_MNIST_Model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('../model/'))  # Automatically get the last saved model

    # restore placeholder variable
    graph = tf.get_default_graph()
    x_input = graph.get_tensor_by_name('x_input:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')

    # restore op
    op_to_predict = graph.get_tensor_by_name('op_to_predict:0')

    for i in range(len(imagearray)):
        # print(i)
        img_norm = img_normal(imagearray[i])
        img_newshape = np.reshape(img_norm, [-1, 28 * 28])

        feed_dict = {x_input: img_newshape, keep_prob: 1.0}
        predint = sess.run(op_to_predict, feed_dict)
        print(predint[0])
        y_Predicted.append(int(predint))

    print("END")


if __name__ == '__main__':
    print("Started")
    detect_all_img()
    print("Length of Array Predicted : ", len(y_Predicted))
    print(y_Predicted)

    # data = asarray([y_Predicted])
    # save('test_predicted.npy', data)
    # print("Saved test_predicted.npy file")