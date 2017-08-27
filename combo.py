import colorfilter as cf
import intensityfeatures as intensity
import Model1 as model
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

import os
import sys
import cv2
import math
import cPickle
import numpy as np
import scipy.misc as sm

imgprots = {}
objprots = {}

with open('imgprots.dat', 'rb') as f:
    imgprots['dir'] = cPickle.load(f)

with open('newprots/colorprots.dat', 'rb') as f:
    imgprots['color'] = cPickle.load(f)

with open('newprots/intensityprots.dat', 'rb') as f:
    imgprots['intensity'] = cPickle.load(f)

with open('newprots/dirobjprots.dat', 'rb') as f:
    objprots['dir'] = cPickle.load(f)#[objidx]

with open('newprots/colorobjprots.dat', 'rb') as f:
    objprots['color'] = cPickle.load(f)#[objidx]

with open('newprots/intensityobjprots.dat', 'rb') as f:
    objprots['intensity'] = cPickle.load(f)#[objidx]

# for o in objprots:
#     print o, len(objprots[o])

def normalize_dict(data):
    smallest = 1e10
    largest = 0
    for k in data:
        smallest = min(np.min(data[k]), smallest)
        largest = max(np.max(data[k]), largest)
    for k in data:
        data[k] = (np.array(data[k]) - smallest)/(largest - smallest)
    return data

# order = ['dir', 'color', 'intensity']
order = ['color',]

def combine_dict(d, mode='np'):
    if mode == 'np':
        combined = np.array([])
        for o in order:
            combined = np.concatenate((combined, d[o]))
    else:
        combined = []
        for o in order:
            combined += d[o]
    return combined

def combomodulation(s2bout, objprots):
    #s2boutputs dimension: numScales x n x n x numProts
    lipMap = []
    for scale in xrange(len(S2boutputs)):
        S2bsum = np.sum(S2boutputs[scale], axis = 2)
        S2bsum = S2bsum[:,:,np.newaxis]
        lip = (S2boutputs[scale] * feedback)/(S2bsum + opt.STRNORMLIP)
        lipMap.append(lip)
    return lipMap

def split_feedback(feedback):
    ret = {}
    for idx, t in enumerate(order):
        ret[t] = feedback[idx*600:(idx+1)*600]
    return ret

def rescale_combine(scales, size):
    ret = np.zeros(size)
    for s in scales:
        lip_s = np.sum(s, axis=2)
        ret += sm.imresize(lip_s, size)
    return ret

# then do same thing for not double opponency
def combo(imgpath, imgidx):

    # objidx = int(sys.argv[2])
    objidx = int(os.path.basename(imgpath).split('.')[0].split('-')[1]) - 1
    # objidx = 1

    print np.argmax(model.feedbackSignal(objprots['color'], objidx))

    s2bout = {}
    c1out = {}
    img_m = sm.imread(imgpath, mode='I')
    img_o = cv2.imread(imgpath)

    def run_dir():
        s1filters = model.buildS1filters()
        print 'dir'
        s1out = model.runS1layer(img_m, s1filters)
        c1out['dir'] = model.runC1layer(s1out)
        s2bout['dir'] = model.runS2blayer(c1out['dir'], imgprots['dir'])

    def run_color():
        print 'color'
        c1out['color'] = cf.runS1C1(img_o)
        s2bout['color'] = model.runS2blayer(cf.groupnormalize(c1out['color']), imgprots['color'])

    def run_intensity():
        print 'intensity'
        c1out['intensity'] = intensity.runS1C1(img_o)
        s2bout['intensity'] = model.runS2blayer(cf.groupnormalize(c1out['intensity']), imgprots['intensity'])

    
    # run_dir()
    run_color()
    # run_intensity()

    objprots_c = combine_dict(objprots, mode='list')
    s2bout_c = combine_dict(s2bout)
    print 'combined'

    # objprots_c[objprots_c == 0] = 9999999999999999999999999999999999999999999999999.0
    # objprots_c[objprots_c == 0] = float('inf')

    # print np.unique(objprots_c)

    # feedback = split_feedback(model.feedbackSignal(objprots_c, objidx))
    feedback = {}
    for o in order:
        feedback[o] = model.feedbackSignal(objprots[o], objidx) #feedback['color'] has nans
    feedback_c = combine_dict(feedback)#, mode='list')
    print 'feedback'    

    # import ipdb;ipdb.set_trace()

    lipmaps = {}
    for o in order:
        lipmaps[o] = model.topdownModulation(s2bout[o], feedback[o], norm=(o != 'color'))
    print 'lipmaps'

    # lipmap = model.topdownModulation(s2bout_c, feedback_c)

    priomaps = {}
    # priomaps['dir'] = model.priorityMap(lipmaps['dir'], [256, 256]) # 12 + 6 + 6
    # then add the other lipmaps
    priomaps['color'] = rescale_combine(lipmaps['color'], [256, 256])
    # priomaps['intensity'] = rescale_combine(lipmaps['intensity'], [256, 256])
    # priorityMap = priomap_dir
    priorityMap = np.zeros([256, 256])
    for p in priomaps:
        priorityMap += priomaps[p]
    print 'priomap'

    # print priomap_dir

    # priorityMap = model.priorityMap(lipmap,[256,256])

    fig,ax = plt.subplots(nrows=1,ncols=2)
    plt.gray()
    pmap, minV, maxV = model.imgDynamicRange(priomaps['color'])
    dims = pmap.shape
    pmap = model.scale(priomaps['color'])
    for i in xrange(dims[0]):
        for j in xrange(dims[0]):
            tmp = pmap[i,j]
            pmap[i,j]= np.exp(tmp)
            # pmap[i,j]= np.exp(np.exp(np.exp(tmp)))
    ax[0].imshow(gaussian_filter(pmap, sigma=3))

    for i in xrange(dims[0]):
        for j in xrange(dims[0]):
            tmp = pmap[i,j]
            pmap[i,j]= np.exp(tmp)
            # pmap[i,j]= np.exp(np.exp(np.exp(tmp)))
    ax[1].imshow(gaussian_filter(pmap, sigma=3))

    # for idx, o in enumerate(order):
    #     pmap, minV, maxV = model.imgDynamicRange(priomaps[o])
    #     dims = pmap.shape
    #     pmap = model.scale(priomaps[o])
    #     for i in xrange(dims[0]):
    #         for j in xrange(dims[0]):
    #             tmp = pmap[i,j]
    #             pmap[i,j]= np.exp(np.exp(tmp))
    #             # pmap[i,j]= np.exp(np.exp(np.exp(tmp)))
    #     ax[idx + 1, 0].imshow(gaussian_filter(pmap, sigma=3))

    #     for i in xrange(dims[0]):
    #         for j in xrange(dims[0]):
    #             tmp = pmap[i,j]
    #             pmap[i,j]= np.exp(tmp)
    #             # pmap[i,j]= np.exp(np.exp(np.exp(tmp)))
    #     ax[idx + 1, 1].imshow(gaussian_filter(pmap, sigma=3))

    for which in ['color']:
        print 'working on {}'.format(which)
        protID = np.argmax(feedback[which])
        numRows = len(c1out[which])
        fig,ax = plt.subplots(nrows=numRows, ncols=4)
        plt.gray()  # show the filtered result in grayscale
        i = 0
        for scale in c1out[which]:
            # if len(scale.shape) == 2:
            #     scale = scale[:,:,np.newaxis]
            s2b, minV, maxV = model.imgDynamicRange(scale[:,:,0])
            ax[i,0].imshow(scale[:,:,0])
            i += 1
        i = 0
        for scale in c1out[which]:
            # if len(scale.shape) == 2:
            #     scale = scale[:,:,np.newaxis]
            s2b, minV, maxV = model.imgDynamicRange(scale[:,:,1])
            ax[i,1].imshow(scale[:,:,1])
            i += 1
        i = 0
        for scale in s2bout[which]:
            #s2b, minV, maxV = model.imgDynamicRange(np.mean(scale, axis = 2))
            if len(scale.shape) == 2:
                scale = scale[:,:,np.newaxis]
            s2b, minV, maxV = model.imgDynamicRange(scale[:,:,protID])
            ax[i,2].imshow(scale[:,:,protID])
            i += 1
        i = 0
        for scale in lipmaps[which]:
            #s2b, minV, maxV = model.imgDynamicRange(np.mean(scale, axis = 2))
            if len(scale.shape) == 2:
                scale = scale[:,:,np.newaxis]
            s2b, minV, maxV = model.imgDynamicRange(scale[:,:,protID])
            ax[i,3].imshow(np.exp(np.exp(s2b)))
            i += 1
        fig.suptitle(which)


    relative_focus = np.argmax(priorityMap)
    fy = int(math.floor(relative_focus/256))
    fx = int(relative_focus % 256)
    cv2.circle(img_o, (fx, fy), 5, (0, 0, 255))
    colors = [
        (0,0,255), #R - dir
        (0,255,0), #G - color
        (255,0,0), #B - intensity
    ]
    for idx, o in enumerate(order):
        relative_focus = np.argmax(priomaps[o])
        fy = int(math.floor(relative_focus/256))
        fx = int(relative_focus % 256)
        cv2.circle(img_o, (fx, fy), 10, colors[idx])

    cv2.imshow('img', img_o)
    # cv2.imwrite('./pink-lightblue/{}.png'.format(imgidx), img_o)
    # print order
    plt.show()
    cv2.waitKey(0)

    # import ipdb;ipdb.set_trace()

combo(sys.argv[1])
# for idx, imgpath in enumerate(os.listdir('./colorscenes')):
#     combo('./colorscenes/{}'.format(imgpath), idx)

# for k in objprots:
#     print k
#     print 'max:', np.max(objprots[k])
#     print 'min:', np.min(objprots[k])
#     print ''

# objprots = normalize_dict(objprots)

# for k in objprots:
#     print k
#     print 'max:', np.max(objprots[k])
#     print 'min:', np.min(objprots[k])
#     print ''