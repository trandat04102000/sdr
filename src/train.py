#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'yan9yu'

import sys
import numpy as np
import cv2

def letterbox(img, new_shape=(25, 25), color=(255, 255, 255), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img


im = cv2.imread('../data/train.png')
im3 = im.copy()

gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

#################      Now finding Contours         ###################

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

samples = np.empty((0, 784), np.float32)
responses = []
keys = [i for i in range(48, 58)]

resize_size = (15, 25)

for cnt in contours:

    if cv2.contourArea(cnt) > 50:
        [x, y, w, h] = cv2.boundingRect(cnt)

        if h > 28:

            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi = thresh[y:y + h, x:x + w]
            r = resize_size[1] / max(h, w)
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            roi = cv2.resize(roi, (int(w*r), int(h*r)), interpolation=interp)
            roi = 255 - roi
            roismall = letterbox(roi, (28, 28))
            roismall[roismall<60] = 0
            cv2.imshow('norm', im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1, 784))
                samples = np.append(samples, sample, 0)
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
responses = np.array(responses, np.float32)
responses = responses.reshape((responses.size, 1))
print ("training complete")

samples = np.float32(samples)
responses = np.float32(responses)

cv2.imwrite("../data/train_result.png", im)
np.savetxt('../data/generalsamples.data', samples)
np.savetxt('../data/generalresponses.data', responses)
