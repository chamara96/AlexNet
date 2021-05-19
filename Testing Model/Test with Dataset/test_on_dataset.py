import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import cv2
import random

import read_model as alex_predict
import datetime

imagefile = "../../MNIST_data_unziped/t10k-images.idx3-ubyte"
imagearray = idx2numpy.convert_from_file(imagefile)

labelfile = "../../MNIST_data_unziped/t10k-labels.idx1-ubyte"
labelarray = idx2numpy.convert_from_file(labelfile)


def img_normal(raw):
    # normalize pixels to 0 and 1.
    # digit is 1, background is 0
    norm = np.array(raw, np.float32)
    norm /= 255.0
    return norm


def predict_digit(val):
    print("Original:", labelarray[val])
    img_norm = img_normal(imagearray[val])
    img_newshape = np.reshape(img_norm, [-1, 28 * 28])
    # start = datetime.datetime.now()
    pre_val = alex_predict.detect_return(img_newshape)
    # end = datetime.datetime.now()
    # print('cost time:%d s' % (end - start).seconds)
    return pre_val


val = random.randint(0, 1000)
print("Random Position:", val)
temp = predict_digit(val)
print("Predicted:", temp)

# cv2.imshow('frame',imagearray[0])
# cv2.waitKey(0)

# plt.imshow(imagearray[1], cmap=plt.cm.binary)

# Plot Letter A

plt.figure(figsize=(3, 3))
plt.title('Original Digit = {}\n Predicted Digit = {}'.format(labelarray[val], temp))
plt.imshow(imagearray[val], cmap=plt.cm.gray)
plt.show()
