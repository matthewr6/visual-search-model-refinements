import sys
import cv2
import os
import math
import random
import numpy as np
import cPickle
import time
import matplotlib.pyplot as plt
import scipy.ndimage.filters as snf
from scipy.ndimage.filters import gaussian_filter, convolve
import Model1 as model

def buildFilter():
    RFSIZE=7
    filtsthissize=[]
    for o in [90, 180]:
        theta = np.radians(o) 
        #print "RF SIZE:", RFSIZE, "orientation: ", theta / math.pi, "* pi"
        x, y = np.mgrid[0:RFSIZE, 0:RFSIZE] - RFSIZE/2
        sigma = 0.0036 * RFSIZE * RFSIZE +0.35 * RFSIZE + 0.18
        lmbda = sigma / 0.8 
        gamma = 0.3
        x2 = x * np.cos(theta) + y * np.sin(theta)
        y2 = -x * np.sin(theta) + y * np.cos(theta)
        myfilt = (np.exp(-(x2*x2 + gamma * gamma * y2 * y2) / (2 * sigma * sigma))
                * np.cos(2*math.pi*x2 / lmbda))
        #print type(myfilt[0,0])
        myfilt[np.sqrt(x**2 + y**2) > (RFSIZE/2)] = 0.0
        # Normalized like in Minjoon Kouh's code
        myfilt = myfilt - np.mean(myfilt)
        myfilt = myfilt / np.sqrt(np.sum(myfilt**2))
        filtsthissize.append(myfilt.astype('float'))
    return filtsthissize

s1filters = model.buildS1filters() # s1 layer is convolution.

basefilter = buildFilter()

def half_square(img):
    img[img < 0] = 0
    return img**2.0

def rectify(img, norm):
    k = 1.0
    top = k * img
    sigma = 0.225
    bottom = sigma**2 + norm
    return np.sqrt(top/bottom)

def normalize(mat, maxv=1.0):
    if np.min(mat) != 0:
        mat = mat - np.min(mat)
    mat = mat / np.max(mat)
    mat = mat * maxv
    return mat

def run_filters(img):
    ret = []
    for direction in basefilter:
        ret.append(cv2.filter2D(img, -1, direction))
    return np.dstack(ret)

def single_opponent(img):
    channels = {
        'r': img[:,:,2], 
        'g': img[:,:,1], 
        'b': img[:,:,0]
    }
    combined = {}
    combinations = ['rg', 'gr', 'by', 'yb',] # how does black/white work
    for combo in combinations:
        a = combo[0]
        b = combo[1]
        if a == 'y':
            first = (channels['r'] + channels['g']) * 0.5
            second = channels[b]
        if b == 'y':
            second = (channels['r'] + channels['g']) * 0.5
            first = channels[a]
        if combo not in ['by', 'yb']:
            first = channels[a]
            second = channels[b]
        combined[combo] = first - second
        # combined[combo] = np.abs(first - second)
        combined[combo][combined[combo] < 0] = 0
        print np.max(combined[combo])

    maxed = {}
    for c in combined:
        maxed[c] = snf.maximum_filter(combined[c], size=9)

    for c in channels:
        if 'r' in c or 'g' in c:
            print np.unique(channels[c])
            cv2.imshow(c, normalize(channels[c]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for c in combined:
        if 'r' in c or 'g' in c:
            cv2.imshow(c, normalize(combined[c]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return combined

def double_opponent(channels):# expect n x n x 4 x 4
    convolved = {}
    for c in channels:
        convolved[c] = run_filters(channels[c])
    combined = {}
    combined['rg'] = cv2.absdiff(convolved['rg'], convolved['gr'])
    combined['by'] = cv2.absdiff(convolved['by'], convolved['yb'])

    for c in convolved:
        cv2.imshow(c, normalize(np.mean(convolved[c],axis=2)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('rg',snf.maximum_filter(normalize(np.mean(combined['rg'],axis=2)), size=9))
    cv2.imshow('by',snf.maximum_filter(normalize(np.mean(combined['by'],axis=2)), size=9))
    cv2.waitKey(0)

if __name__ == '__main__':
    so = single_opponent(cv2.imread(sys.argv[1]))
    # do = double_opponent(so)