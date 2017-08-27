import cPickle
import colorfilter as cf
import sys
import cv2
import Model1 as model
import random
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
random.seed(datetime.now())

beginning = 372
change = 10

targetidx = int(sys.argv[2])

# Build filters
with open('colorprots.dat', 'rb') as f:
    imgprots = cPickle.load(f)
with open('colorobjprots.dat', 'rb') as f:
    objprots = cPickle.load(f)

img = cv2.imread(sys.argv[1])
# img = cv2.resize(img, None, fx=0.5, fy=0.5)
c1out = cf.runS1C1(img)
s2bout = model.runS2blayer(cf.groupnormalize(c1out), imgprots) # 6 x n x n x 600
feedback = model.feedbackSignal(objprots, targetidx)
# print feedback
lipmap = model.topdownModulation(s2bout,feedback)
protidx = np.argmax(feedback)

priorityMap = model.priorityMap(lipmap,[256,256])

nrows = 6
fig, ax = plt.subplots(nrows=nrows, ncols=5)
plt.gray()
for i in range(nrows):
    ax[i, 0].imshow(c1out[i][:,:,0])
    ax[i, 1].imshow(c1out[i][:,:,1])
    ax[i, 2].imshow(np.mean(c1out[i], axis=2))
    ax[i, 3].imshow(s2bout[i][:,:,protidx])
    ax[i, 4].imshow(lipmap[i][:,:,protidx])

fig,ax = plt.subplots(nrows = 1, ncols = 2)
plt.gray()
dims = priorityMap.shape
pmap = model.scale(priorityMap)
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

plt.show()