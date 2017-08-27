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
from scipy.ndimage.filters import gaussian_filter
import Model1 as model

def buildFilter():
    RFSIZE=3
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

def normalize(mat, maxv=255.0):
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
    # print img.shape
    # cv2.imshow('a', img)
    # cv2.waitKey(0)
    channels = {
        'r': img[:,:,2], 
        'g': img[:,:,1], 
        'b': img[:,:,0]
    }

    convolved = {}
    for c in channels:
        # convolved[c] = model.runS1layer(channels[c], [s1filters])[0] # this is bad do not do this... maybe? unsure.
        convolved[c] = run_filters(channels[c])
        # pimg = normalize(np.mean(convolved[c], axis=2).astype(float))
        # cv2.imshow(c, cv2.resize(pimg,(256, 256)))
        # also two directions or four?
    # cv2.waitKey(0)
    combined = {}
    combinations = ['rg', 'gr', 'by', 'yb',] # how does black/white work
    for combo in combinations:
        a = combo[0]
        b = combo[1]
        # combo is b/y- y is r + g, b is b
        if a == 'y':
            first = (convolved['r'] + convolved['g']) * (1/np.sqrt(6))
            second = convolved[b] * 2/np.sqrt(6)
        if b == 'y':
            second = (convolved['r'] + convolved['g']) * (1/np.sqrt(6))
            first = convolved[a] * 2/np.sqrt(6)
        if combo not in ['by', 'yb']:
            first = convolved[a]
            second = convolved[b]
        if a in 'by':
            combined[combo] = first - second
        else:
            # print np.unique(first - second)
            # print np.unique(first)
            # print np.unique(second)
            combined[combo] = (first - second) * (1/np.sqrt(2))
        # print np.min(combined[combo])
    rectified = {}
    full_array = np.stack([half_square(combined[a]) for a in combined], axis=3) # img size x img size x orientations x channels
    # # print np.sum(full_array, axis=3).shape
    for combo in combined:
        rectified[combo] = rectify(half_square(combined[combo]), np.sum(full_array, axis=3))

    # for c in convolved:
    #     pimg = normalize(np.mean(convolved[c], axis=2))
    #     cv2.imshow(c, cv2.resize(pimg, (256, 256)))

    for c in channels:
        pimg = (channels[c])
        # cv2.imshow(c, cv2.resize(pimg, (256, 256)))
        # plt.imshow(pimg)
        # plt.show()
    # cv2.imshow('a', ((channels['r'] - channels['g'])))
    
    fig,ax = plt.subplots(ncols=4,nrows=3)
    plt.gray()
    ax[0,0].imshow(channels['r'])
    ax[0,1].imshow(channels['g'])
    ax[0,2].imshow(channels['b'])
    ax[0,3].imshow(0.5 * (channels['g'] + channels['r']))
    ax[1,0].imshow(np.mean(convolved['r'],axis=2))
    ax[1,1].imshow(np.mean(convolved['g'],axis=2))
    ax[1,2].imshow(np.mean(convolved['b'],axis=2))
    ax[1,3].imshow(0.5 * (np.mean(convolved['g'],axis=2) + np.mean(convolved['r'],axis=2)))
    ax[2,0].imshow(np.mean(rectified['rg'],axis=2))
    ax[2,1].imshow(np.mean(rectified['gr'],axis=2))
    ax[2,2].imshow(np.mean(rectified['by'],axis=2))
    ax[2,3].imshow(np.mean(rectified['yb'],axis=2))
    plt.show()

    # plt.imshow(channels['r'] - channels['g'])
    # plt.show()

    cv2.imshow('r', np.sum(convolved['r'],axis=2).astype(float))
    cv2.imshow('g', np.sum(convolved['g'],axis=2).astype(float))
    cv2.imshow('rg', np.sum(convolved['r'] - convolved['g'],axis=2).astype(float))
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # cv2.waitKey(0)
    return rectified

def double_opponent(channels):# expect n x n x 4 x 4
    convolved = {}
    for c in channels:
        convolved[c] = run_filters(channels[c]) # figure this out
    #     cv2.imshow(c, np.mean(convolved[c], axis=2))
    # cv2.waitKey(0)
    rectified = {}
    full_array = np.stack([half_square(convolved[c]) for c in convolved], axis=3)
    for c in convolved:
        rectified[c] = rectify(half_square(convolved[c]), np.sum(full_array, axis=3))
    combined = {}
    combined['rg'] = np.sum(rectified['rg'] + rectified['gr'], axis=2)
    combined['by'] = np.sum(rectified['by'] + rectified['yb'], axis=2)
    # cv2.imshow('rg',combined['rg'])
    # cv2.imshow('by',combined['by'])
    # cv2.waitKey(0)


if __name__ == '__main__':
    so = single_opponent(cv2.imread(sys.argv[1]))
    do = double_opponent(so)