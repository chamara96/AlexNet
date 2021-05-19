# import the necessary packages
import numpy as np
import cv2
import tensorflow as tf

# from test import show_output

y_Predicted = []


def show_output(predicted_arr):
    image = cv2.imread('Data/bank_slip_croped.jpg')

    grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
    array_qq = np.zeros((40, 347, 3), dtype="uint8")
    numpy_vertical = np.vstack((image, array_qq))

    text = " ".join(str(x) for x in predicted_arr)

    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (10, 70)
    fontScale = 0.63
    color = (255, 0, 0)
    thickness = 2

    image_txt = cv2.putText(numpy_vertical, text, org, font, fontScale,
                            color, thickness, cv2.LINE_AA, False)
    cv2.imshow('Bank Slip Digits', image_txt)
    cv2.waitKey(0)


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

    for i in range(15):
        # print(i + 1)
        image = cv2.imread("Data/decoded_digits/" + str(i + 1) + ".png")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.GaussianBlur(gray, (3, 3), 0)
        ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)  # 175
        img_rez = cv2.resize(thresh, (28, 28))
        # cv2.imshow("Gray", img_rez)
        # cv2.waitKey(0)

        # print(i)
        img_norm = img_normal(img_rez)
        img_newshape = np.reshape(img_norm, [-1, 28 * 28])

        feed_dict = {x_input: img_newshape, keep_prob: 1.0}
        predint = sess.run(op_to_predict, feed_dict)
        # predict = tf.argmax(feed_dict, dimension=1)
        print(predint[0])
        y_Predicted.append(int(predint))

    print("END")


if __name__ == '__main__':
    print("Started")
    detect_all_img()
    print("Length of Array Predicted : ", len(y_Predicted))
    print(y_Predicted)
    show_output(y_Predicted)
