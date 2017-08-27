import traceback
import sys
import cPickle
import Model1
import scipy.misc
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import ModelOptions1 as opt


reload(opt)
reload(Model1) 

beginning = 372
change = 10


# Build filters
s1filters = Model1.buildS1filters()
print 'Loaded s1 filters'
protsfile = open('imgprots.dat', 'rb')
imgprots = cPickle.load(protsfile)#[beginning:beginning+change]
print 'Loading objprots filters'
protsfile = open('objprotsCorrect.dat', 'rb')
# protsfile = open('resizedobjprots.dat', 'rb')
objprots = cPickle.load(protsfile)
for idx, _ in enumerate(objprots):
    objprots[idx] = objprots[idx]#[beginning:beginning+change]
# objprots = objprots[0:-1] # NOTE THIS IS HACK because objprots was generated from a folder with 41 instead of 40 images. Getting rid of the last img.
print 'Objprots shape:', len(objprots), objprots[0].shape
protsfile = open('naturalImgC2b.dat', 'rb')
imgC2b = cPickle.load(protsfile)
print 'imgC2b: ', len(imgC2b)
imgC2b = imgC2b[0:-1]
with open('S3prots.dat', 'rb') as f:
# with open('resizeds3prots.dat', 'rb') as f:
    s3prots = cPickle.load(f)[:-1]
#num_objs x num_scales x n x n x prototypes

# Model1.buildS3Prots(1720,s1filters,imgprots)

#num_scales x n x n x prototypes

objNames = Model1.getObjNames()

# #objects
hat = 0
butterfly=13
binoculars = 8
tuba = 31
ant = 3
camera = 14
statue = 12
fan = 16
phonograph = 36
piano = 37
spiral = 4
lobster = 22
accordion = 1
turtle = 38

targetIndex = hat
stimnum = 4
location = (0, 2) # ZERO INDEXED

scaleSize = 8
protID = 0
print 'Obj names', objNames[targetIndex]

def check_bounds(x, y):
    wh = 256/3.0
    bounds = [
        location[0] * wh,
        (location[0]+1) * wh,
        location[1] * wh,
        (location[1]+1) * wh
    ]
    print x, y, bounds
    return x >= bounds[0] and x <= bounds[1] and y >= bounds[2] and y <= bounds[3]

img = scipy.misc.imread('example.png')
# import cv2
# img = cv2.imread('example.png')
# img = scipy.misc.imread('hatonly.png', mode='I')
# img = scipy.misc.imread('objectimages/1.normal.png')
# img = scipy.misc.imread('stimuli/1.array{}.ot.png'.format(stimnum))
S1outputs = Model1.runS1layer(img, s1filters)
C1outputs = Model1.runC1layer(S1outputs)
S2boutputs = Model1.runS2blayer(C1outputs, imgprots)
feedback = Model1.feedbackSignal(objprots, targetIndex)
print 'feedback info: ', feedback.shape
lipmap = Model1.topdownModulation(S2boutputs,feedback)
protID = np.argmax(feedback)
print feedback[protID], np.mean(feedback)
with open('sample_unmodified_lip.dat', 'wb') as f:
    cPickle.dump(lipmap, f, protocol=-1)
print 'lipmap shape: ', len(lipmap), lipmap[0].shape
#sif, minV, maxV = Model1.imgDynamicRange(np.mean(S1outputs[scaleSize], axis = 2))
#print 'Sif: ', sif.shape, 'Max: ', maxV, 'Min: ', minV

#cif, minV, maxV = Model1.imgDynamicRange(np.mean(C1outputs[scaleSize], axis = 2))
#print 'Cif: ', cif.shape, 'Max: ', maxV, 'Min: ', minV

#s2b, minV, maxV = Model1.imgDynamicRange(S2boutputs[scaleSize][:,:,protID])
#print 's2b: ', s2b.shape, 'Max: ', maxV, 'Min: ', minV

# C2boutputs = Model1.runC1layer(S2boutputs)
#lm, minV, maxV = Model1.imgDynamicRange(lipmap[scaleSize][:,:,protID])
#print 'lipmap: ', lm.shape, 'Max: ', maxV, 'Min: ', minV

priorityMap = Model1.priorityMap(lipmap,[256,256])

# i = 0
# found = False
# while i < 5 and not found:
#     priorityMap, fx, fy = Model1.inhibitionOfReturn(priorityMap)
#     found = check_bounds(fx, fy)
#     i += 1

# print i, found

# inhibitions = 1
# for i in xrange(inhibitions):
#     priorityMap = Model1.inhibitionOfReturn(priorityMap)

modulated_s2boutputs = Model1.prio_modulation(priorityMap, S2boutputs)
# cropped_s2boutputs = Model1.crop_s2boutputs(modulated_s2boutputs, priorityMap)
cropped_s2boutputs = modulated_s2boutputs
t = Model1.runS3layer(cropped_s2boutputs, s3prots, priorityMap)
# print t
t2 = Model1.runC3layer(t)
print 'predicted: ', t2
print stimnum, 'stimnum'
print 'is: ', targetIndex

# print t2
# priorityMap = Model1.inhibitionOfReturn(priorityMap)

print 'Feedback signal shape: ', feedback.shape

numCols = 5
numRows = 12

whichgraph = 'ab'


if 'a' in whichgraph:
    fig,ax = plt.subplots(nrows = numRows, ncols = numCols)
    plt.gray()  # show the filtered result in grayscale


    for i in xrange(numRows):
        ax[i,0].imshow(img)

    i = 0
    for scale in S1outputs:
        sif, minV, maxV = Model1.imgDynamicRange(np.mean(scale, axis = 2))
        ax[i,1].imshow(sif)
        i += 1

    i = 0
    for scale in C1outputs:
        cif, minV, maxV = Model1.imgDynamicRange(np.mean(scale, axis = 2))
        ax[i,2].imshow(cif)
        i += 1

    i = 0
    for scale in S2boutputs:
        #s2b, minV, maxV = Model1.imgDynamicRange(np.mean(scale, axis = 2))
        s2b, minV, maxV = Model1.imgDynamicRange(scale[:,:,protID])
        ax[i,3].imshow(s2b)
        i += 1

    i = 0
    for scale in lipmap:
        #lm, minV, maxV = Model1.imgDynamicRange(np.mean(scale, axis = 2))
        lm, minV, maxV = Model1.imgDynamicRange(scale[:,:,protID])  
        ax[i,4].imshow(lm)
        i += 1

    ax[0,0].set_title('Original')
    ax[0,1].set_title('S1')
    ax[0,2].set_title('C1')
    ax[0,3].set_title('S2b')
    ax[0,4].set_title('LIP')

if 'b' in whichgraph:

    fig,ax = plt.subplots(nrows = 1, ncols = 2)
    plt.gray()
    pmap, minV, maxV = Model1.imgDynamicRange(priorityMap)
    dims = pmap.shape
    pmap = Model1.scale(priorityMap)
    for i in xrange(dims[0]):
        for j in xrange(dims[0]):
            tmp = pmap[i,j]
            pmap[i,j]= np.exp(np.exp(tmp))
            # pmap[i,j]= np.exp(np.exp(np.exp(tmp)))
    ax[0].imshow(gaussian_filter(pmap, sigma=3))

    for i in xrange(dims[0]):
        for j in xrange(dims[0]):
            tmp = pmap[i,j]
            pmap[i,j]= np.exp(tmp)
            # pmap[i,j]= np.exp(np.exp(np.exp(tmp)))
    ax[1].imshow(gaussian_filter(pmap, sigma=3))

if 'c' in whichgraph:

    fig,ax = plt.subplots(nrows = numRows, ncols = change)
    plt.gray()  # show the filtered result in grayscale

    for i in xrange(change):
        for j, scale in enumerate(S2boutputs):
            s2b, minV, maxV = Model1.imgDynamicRange(scale[:,:,i])
            ax[j,i].imshow(s2b)


if 'd' in whichgraph:

    fig,ax = plt.subplots(nrows = numRows, ncols = change)
    plt.gray()  # show the filtered result in grayscale
    plt.axis('off')

    for i in xrange(change):
        for j, scale in enumerate(lipmap):
            s2b, minV, maxV = Model1.imgDynamicRange(scale[:,:,i])
            ax[j,i].imshow(s2b)

if 'e' in whichgraph:
    fig,ax = plt.subplots(nrows = numRows, ncols = 3)
    plt.gray()  # show the filtered result in grayscale
    i = 0
    for scale in S2boutputs:
        #s2b, minV, maxV = Model1.imgDynamicRange(np.mean(scale, axis = 2))
        s2b, minV, maxV = Model1.imgDynamicRange(scale[:,:,protID])
        ax[i,0].imshow(np.exp(np.exp(s2b)))
        i += 1
    i = 0
    for scale in modulated_s2boutputs:
        #s2b, minV, maxV = Model1.imgDynamicRange(np.mean(scale, axis = 2))
        s2b, minV, maxV = Model1.imgDynamicRange(scale[:,:,protID])
        ax[i,1].imshow(np.exp(np.exp(s2b)))
        i += 1
    i = 0
    for scale in cropped_s2boutputs :
        #s2b, minV, maxV = Model1.imgDynamicRange(np.mean(scale, axis = 2))
        s2b, minV, maxV = Model1.imgDynamicRange(scale[:,:,protID])
        ax[i,2].imshow(np.exp(np.exp(s2b)))
        i += 1


plt.show()

# protsfile = open('naturalImgC2b.dat', 'wb')
# try:
#   prots = Model1.buildObjProts(s1filters, imgprots)
#   cPickle.dump(prots, protsfile, protocol = -1)
# except: # Exception as e:
#   tb = traceback.format_exc()
#   print tb
