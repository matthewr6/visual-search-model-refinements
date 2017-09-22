import traceback
import sys
import cPickle
import Model1 as model
import scipy.misc
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import ModelOptions1 as opt
import json
import os

import colormodel

def save_priomap(prio, name):
    pmap = np.exp(np.exp(np.exp(model.scale(prio))))
    # plt.imshow(gaussian_filter(pmap, sigma=3), cmap='hot')
    plt.imsave('{}.png'.format(name), gaussian_filter(pmap, sigma=3), format='png', cmap='hot')

# with open('sample_priomap.dat', 'rb') as f:
#     priorityMap = cPickle.load(f)


# with open('sample_lipmaps.dat', 'rb') as f:
    # lipmaps = cPickle.load(f)

targetidx = int(sys.argv[1])
with open('./prots/comboobjprots_separatefeatures.dat', 'rb') as f:
    objprots = cPickle.load(f)
feedback, fmeans = colormodel.feedbackSignals(objprots, targetidx)

with open('s2bout.dat', 'rb') as f:
    s2bout = cPickle.load(f)

lipmaps = colormodel.topdownModulation(s2bout, feedback)

priorityMap, individualPriorityMaps = colormodel.comboPriorityMap(lipmaps, [256,256], fmeans)

def normalize(mat, maxv=1.0, lowest=None, highest=None):
    if lowest:
        mat = mat - lowest
    else:
        mat = mat - np.min(mat)
    if highest:
        assert lowest is not None
        mat = mat / (highest - lowest)
    else:
        mat = mat / np.max(mat)
    mat = mat * maxv
    return mat

name = 'bw'
path = '../paper-colormodel/images/priomaps/{}'.format(name)

pmap = np.exp(np.exp(np.exp(normalize(priorityMap))))
plt.axis('off')
fig = plt.imshow(gaussian_filter(pmap, sigma=3), cmap='hot')
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
print 'original'
plt.savefig('{}_final.png'.format(path), bbox_inches='tight', pad_inches = 0)
i = 1
for k in individualPriorityMaps:
    pmap = np.exp(np.exp(np.exp(normalize(individualPriorityMaps[k]))))
    fig = plt.imshow(gaussian_filter(pmap, sigma=3), cmap='hot')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    print k
    plt.savefig('{}_{}'.format(path, k), bbox_inches='tight', pad_inches = 0)
    i += 1

# inhibitions = 3
# displays = [gaussian_filter(priorityMap,sigma=3)]
# for i in xrange(inhibitions):
#     print np.max(displays[-1])
#     n = model.inhibitionOfReturn(displays[-1])
#     print n[1:]
#     displays.append(normalize(n[0]))

# fig,ax = plt.subplots(nrows = len(displays),ncols=2)
# plt.gray()
# # plt.pcolor(gaussian_filter(np.exp(np.exp(np.exp(priorityMap))), sigma=3), cmap='hot')
# for idx, pmap in enumerate(displays):
#     pmap = np.exp(np.exp(model.scale(pmap)))
#     ax[idx,0].imshow(gaussian_filter(pmap, sigma=3))#, cmap='hot')

# for idx, pmap in enumerate(displays):
#     pmap = np.exp(np.exp(np.exp(model.scale(pmap))))
#     ax[idx,1].imshow(gaussian_filter(pmap, sigma=3))#, cmap='hot')

# plt.show()