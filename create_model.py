import sys

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

train_num = mnist.train.images.shape[0]

########## set net hyperparameters ##########
learning_rate = 0.0005
batch_size = 100
display_step = train_num / batch_size


epoch_no = 10
training_epochs = int(epoch_no * train_num / batch_size)

n_input = 28 * 28  # image size of input
n_classes = 10  # label class
dropout = 0.9

x_input = tf.placeholder(tf.float32, [None, n_input], name='x_input')
print("======X-INPUT ======")
y_output = tf.placeholder(tf.float32, [None, n_classes], name='y_output')
print("======Y-OUTPUT ======")
print(y_output)
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
print("======KEEP PROB ======")
print(keep_prob)


def conv2d(_input, filter, _biases, name):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(_input, filter, strides=[1, 1, 1, 1], padding='SAME'), _biases),
                      name=name)


def max_pool(_input, k_size, k_strides, name):
    return tf.nn.max_pool(_input, ksize=[1, k_size, k_size, 1], strides=[1, k_strides, k_strides, 1], padding='SAME',
                          name=name)


def norm(_input, lsize=4, name=''):
    return tf.nn.lrn(_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def fc(_input, _weight, _biases, name):
    return tf.nn.relu(tf.nn.xw_plus_b(_input, _weight, _biases), name=name)


def alex_net(_input, _weights, _biases, _dropput):
    # reshape
    input = tf.reshape(_input, [-1, 28, 28, 1])
    print("++++++++++++++")

    #####   Layer1
    # conv
    conv1 = conv2d(input, _weights['wc1'], _biases['bc1'], 'conv1')
    # pool
    pool1 = max_pool(conv1, 2, 2, 'pool1')
    # lrn
    norm1 = norm(pool1, 4, 'lrn1')
    # dropout
    norm1 = tf.nn.dropout(norm1, _dropput)

    #####   Layer2
    # conv
    conv2 = conv2d(norm1, _weights['wc2'], _biases['bc2'], 'conv2')
    # pool
    pool2 = max_pool(conv2, 2, 2, 'pool2')
    # lrn
    norm2 = norm(pool2, 4, 'lrn2')
    # dropout
    norm2 = tf.nn.dropout(norm2, _dropput)

    #####   Layer3
    # conv
    conv3 = conv2d(norm2, _weights['wc3'], _biases['bc3'], 'conv3')

    #####   Layer4
    # conv
    conv4 = conv2d(conv3, _weights['wc4'], _biases['bc4'], 'conv4')

    #####   Layer5
    # conv
    conv5 = conv2d(conv3, _weights['wc5'], _biases['bc5'], 'conv5')

    # pool
    pool3 = max_pool(conv5, 2, 2, 'pool3')
    # lrn
    norm3 = norm(pool3, 4, 'lrn3')
    # dropout
    norm3 = tf.nn.dropout(norm3, _dropput)

    #####   layer6
    # fc
    fc1 = tf.reshape(norm3, [-1, _weights['wf1'].get_shape().as_list()[0]])

    fc1 = fc(fc1, _weights['wf1'], _biases['bf1'], 'fc1')
    #####   Layer7
    # fc
    fc2 = fc(fc1, _weights['wf2'], _biases['bf2'], 'fc2')

    #####   Output
    out = tf.nn.xw_plus_b(fc2, _weights['out'], _biases['out'])

    return out


weights = {
    'wc1': tf.Variable(tf.random_normal([11, 11, 1, 96]), name='wc1'),
    'wc2': tf.Variable(tf.random_normal([5, 5, 96, 256]), name='wc2'),
    'wc3': tf.Variable(tf.random_normal([3, 3, 256, 384]), name='wc3'),
    'wc4': tf.Variable(tf.random_normal([3, 3, 384, 384]), name='wc4'),
    'wc5': tf.Variable(tf.random_normal([3, 3, 384, 256]), name='wc5'),

    'wf1': tf.Variable(tf.random_normal([4 * 4 * 256, 1024]), name='wf1'),
    'wf2': tf.Variable(tf.random_normal([1024, 200]), name='wf2'),
    'out': tf.Variable(tf.random_normal([200, 10]), name='w_output')
}
biases = {
    'bc1': tf.Variable(tf.random_normal([96]), name='bc1'),
    'bc2': tf.Variable(tf.random_normal([256]), name='bc2'),
    'bc3': tf.Variable(tf.random_normal([384]), name='bc3'),
    'bc4': tf.Variable(tf.random_normal([384]), name='bc4'),
    'bc5': tf.Variable(tf.random_normal([256]), name='bc5'),

    'bf1': tf.Variable(tf.random_normal([1024]), name='bf1'),
    'bf2': tf.Variable(tf.random_normal([200]), name='bf2'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='b_output')
}

# construct model
predict = alex_net(x_input, weights, biases, keep_prob)

predint = tf.argmax(predict, 1, 'op_to_predict')

# loss
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=predict, labels=y_output, name='op_to_loss'))
# train
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
# test
correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(y_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='op_to_accuracy')
# init variables
init = tf.initialize_all_variables()
# create saver
saver = tf.train.Saver()
# train begin

accuracy_plt = []
loss_plt = []

with tf.Session() as sess:
    sess.run(init)
    for i in range(training_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        if i % display_step == 0:
            loss_plt_temp, accuracy_plt_temp = sess.run([cross_entropy, accuracy],
                                                        feed_dict={x_input: batch_xs, y_output: batch_ys,
                                                                   keep_prob: 1.0})

            print(loss_plt_temp, "===", accuracy_plt_temp)
            accuracy_plt.append(accuracy_plt_temp)
            loss_plt.append(loss_plt_temp)
            # print('step:%d,training accuracy:%g' % (i, sess.run([cross_entropy, accuracy],
            #                                                     feed_dict={x_input: batch_xs, y_output: batch_ys,
            #                                                                keep_prob: 1.0})))
        sess.run(train_step, feed_dict={x_input: batch_xs, y_output: batch_ys, keep_prob: dropout})
    saver.save(sess, 'model/AlexNet_MNIST_Model.ckpt')  ##Save model files

    print('Optimization Finished!')
    # test accuracy
    print("Testing Accuracy:%g" % sess.run(accuracy, feed_dict={x_input: mnist.test.images,
                                                                y_output: mnist.test.labels, keep_prob: 1.}))