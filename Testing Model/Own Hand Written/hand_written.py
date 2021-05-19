# coding:utf-8
import cv2 as cv
import numpy as np
import sys
import datetime
import read_model as alex_predict

width = 512
height = 512
img = 0
palette = 0
point = (-1, -1)
min_x = sys.maxsize
min_y = sys.maxsize
max_x = -1
max_y = -1
line_thickness1 = 3
line_thickness2 = 20
drawing = False


def img_normal(raw):
    # normalize pixels to 0 and 1.
    # digit is 1, background is 0
    norm = np.array(raw, np.float32)
    norm /= 255.0
    return norm


def palette_init(width, height, channels, type, border=False):
    palette = np.zeros((width, height, channels), type)
    if border == True:
        cv.line(palette, (0, int(height / 2)), (width - 1, int(height / 2)), (255, 255, 255), 1)
        cv.line(palette, (int(width / 2), 0), (int(width / 2), height - 1), (255, 255, 255), 1)
    return palette


def detect2():
    # process
    # roi = cv.bitwise_not(roi)
    # cv.imshow('roi', img)

    img_capture = cv.resize(img, (28, 28))
    cv.imshow('img_capture', img_capture)

    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    img_capture = cv.dilate(img_capture, kernel)
    # cv.imshow('dilate', img_capture)

    img_norm = img_normal(img_capture)
    # img_newshape = np.reshape(img_norm,[-1,28,28,1])
    # print(img_newshape.shape)

    img_newshape = np.reshape(img_norm, [-1, 28 * 28])
    start = datetime.datetime.now()
    alex_predict.detect(img_newshape)
    # cnn.forward(img_newshape)
    end = datetime.datetime.now()
    print('cost time:%d s' % (end - start).seconds)


if __name__ == '__main__':

    while (1):
        image = cv.imread("sample_hand_written.jpg")
        cv.imshow('Original', image)
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        img = cv.threshold(gray, 120, 255, cv.THRESH_BINARY_INV)[1]

        # ret, thresh = cv.threshold(img_gau, 115, 255, cv.THRESH_BINARY_INV)
        cv.imshow('Gray scaled', img)

        c = cv.waitKey(33) & 0xFF
        if c == ord('q'):
            cv.destroyAllWindows()
            break;
        elif c == ord('r'):
            palette = palette_init(512, 512, 1, np.uint8, border=True)
            img = palette_init(512, 512, 1, np.uint8)
            min_x = sys.maxsize
            min_y = sys.maxsize
            max_x = -1
            max_y = -1
        elif c == ord('f'):
            detect2()
