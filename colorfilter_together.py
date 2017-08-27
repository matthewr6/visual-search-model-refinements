import sys
import cv2
import os
import random
import numpy as np
import cPickle
import time
import matplotlib.pyplot as plt
import scipy.ndimage.filters as snf
from scipy.ndimage.filters import gaussian_filter
import Model1 as model

s1filters = model.buildS1filters()

def parse_to_show(img, shape):
    return cv2.resize(model.imgDynamicRange(np.mean(img, axis=2))[0], shape[:-1])

def show_imgs(imgs, i, shape):
    cv2.imshow('L', parse_to_show(imgs[0][i], shape))
    cv2.imshow('a', parse_to_show(imgs[1][i], shape))
    cv2.imshow('b', parse_to_show(imgs[2][i], shape))
    cv2.waitKey(0)

# takes in 3 x 12 x n x n x 4 and turns into 12 x n x n x 12
def transform_c1out(c1out):
    ret = []
    for channel in c1out:
        # channel is 12 x n x n x 4
        for idx, scale in enumerate(channel): # iterates 12 times, scale is n x n x 4 nparray
            if len(ret) <= idx:
                ret.append(scale)
            else: # ret[idx] exists
                ret[idx] = np.dstack((ret[idx],scale))
    return ret

# def runS1C1(imgpath):
#     img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_RGB2Lab)
#     shape = img.shape
#     img = [img[:,:,0], img[:,:,1], img[:,:,2]]
#     s1out = []
#     for i in img:
#         s1out.append(model.runS1layer(i, s1filters))
#     c1out = []
#     for s in s1out:
#         c1out.append(model.runC1layer(s))
#     # at this stage c1out is 3 x 12 x n x n x 4 - so now we do each of axis 0 separately or together?
#     # probably together. color + orientation.  12 x n x n x 12
#     return transform_c1out(c1out)
#     # for i in range(12):
#     #     show_imgs(c1out, i, shape)

imgdir = './colorrandoms'
objdir = './complexcolorobjs'
def buildImageProts(numProts): 
    print 'Building ', numProts, 'protoypes from natural images'
    imgfiles = os.listdir(imgdir)
    prots = []
    for n in range(numProts):
        selectedImg = random.choice(range(len(imgfiles)))
        print '----------------------------------------------------'
        
        imgfile = imgfiles[selectedImg]
        print 'Prot number', n, 'select image: ', selectedImg, imgfile
        
        if(imgfile == '._.DS_Store' or imgfile == '.DS_Store'):
            selectedImg = random.choice(range(len(imgfiles)))
            imgfile = imgfiles[selectedImg]
        c1out = runS1C1(imgdir+'/'+imgfile)
        prots.append(model.extract3DPatch(c1out, nbkeptweights=300)) # 100 x (12/4) = 300
    return prots

def buildObjProts(imgProts, resize=True): #computing C2b
    print 'Building object protoypes' 
    imgfiles = os.listdir(objdir) #changed IMAGESFOROBJPROTS to IMAGESFORPROTS
    print imgfiles
    prots = [0 for i in range(len(imgfiles))]
    print 'Prots length: ', len(prots)
    for n in range(len(imgfiles)):
        print '----------------------------------------------------'
        print 'Working on object number', n, ' ', imgfiles[n]
        imgfile = imgfiles[n]
        if(imgfile == '.DS_Store' or imgfile == '._.DS_Store' or imgfile == '._1.normal.png' ):
            continue
        tmp = imgfile.strip().split('.')
        pnum = int(tmp[0]) - 1
        print 'pnum: ', pnum        

        # img = sm.imread(objdir +'/'+imgfile, mode='I') # changed IMAGESFOROBJPROTS to get 250 nat images c2b vals
        # if resize:
        #     img = sm.imresize(img, (64, 64))
        
        t = time.time()
        C1outputs = runS1C1(objdir +'/'+imgfile)

        S2boutputs = model.runS2blayer(C1outputs, imgProts)
        #compute max for a given scale
        max_acts = [np.max(scale.reshape(scale.shape[0]*scale.shape[1],scale.shape[2]),axis=0) for scale in S2boutputs]
        C2boutputs = np.max(np.asarray(max_acts),axis=0) #glabal maximums
        #C2boutputs = runC1layer(S2boutputs)
        prots[pnum] = C2boutputs
        timeF = (time.time()-t)
        print "Time elapsed: ", timeF, " Estimated time of completion: ", timeF*(len(imgfiles)-(n+1))
    return prots

def normalize(mat, maxv=1.0):
    mat = mat - np.min(mat)
    mat = mat / np.max(mat)
    mat = mat * maxv
    return mat

def single_opponents(cielab_img):
    so = {}
    order = ['wb', 'rg', 'yb']
    for i in range(3):
        so[order[i]] = cielab_img[:,:,i]
        so[order[i][::-1]] = 255 - cielab_img[:,:,i]
    return so

def double_opponents(so):
    do = {}
    order = ['wb', 'rg', 'yb']
    for o in order:
        do[o] = cv2.absdiff(so[o] , so[o[::-1]])
    return do

so_order = ['wb','bw','rg','gr','yb','by']
def runS1C1(imgpath):
    img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_RGB2Lab)
    so = single_opponents(img) # six
    do = double_opponents(so) # three, also run filters on these.
    do_filtered = {}
    for d in do:
        do_filtered[d] = model.runS1layer(do[d], s1filters)
    sizes = [i.shape[:2] for i in do_filtered['wb']]
    so_resized = {}
    for s in so:
        so_resized[s] = [cv2.resize(so[s], size) for size in sizes] # each is 12 x n x n
    do_flattened = transform_c1out([do_filtered[d] for d in do_filtered]) # 12 x n x n x 12
    so_flattened = transform_c1out([so_resized[s] for s in so_resized]) # 12 x n x n x 6
    s1out = []
    for i in range(12):
        s1out.append(np.dstack((do_flattened[i], so_flattened[i])))
    c1out = model.runC1layer(s1out)


if __name__ == '__main__':
    modes = ['run', 'show', 'imgprots', 'objprots', 'batch']
    which = 'show'
    assert which in modes

    # so now we have L, a, b - now just run S1 and C1 from the model on them.
    if which == 'show':
        img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_RGB2Lab)
        import time
        t = time.time()
        runS1C1(sys.argv[1])
        print time.time() - t
        # so = single_opponents(img)
        # for d in so:
        #     cv2.imshow(d, so[d])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # do = double_opponents(so)
        # for d in do:
        #     cv2.imshow(d, do[d])
        # cv2.waitKey(0)

    if which == 'batch':
        paths = os.listdir('./colorscenes')
        with open('comboimgprots.dat', 'rb') as f:
            imgprots = cPickle.load(f)
        with open('comboobjprots.dat', 'rb') as f:
            objprots = cPickle.load(f)
        for p in paths:
            targetidx = int(p.split('-')[1]) - 1
            c1out = runS1C1('./colorscenes/{}'.format(p))
            print 'c1out done'
            s2bout = model.runS2blayer(c1out, imgprots)
            print 's2b done'
            feedback = model.feedbackSignal(objprots, targetidx)
            protidx = np.argmax(feedback)
            print 'feedback done'
            lipmap = model.topdownModulation(s2bout, feedback)
            print 'lipmap done'
            priorityMap = model.priorityMap(lipmap, [256,256])
            print priorityMap.shape

            fix, ax = plt.subplots(ncols=3)
            plt.gray()
            pmap = np.exp(np.exp(normalize(priorityMap)))
            ax[0].imshow(gaussian_filter(pmap, sigma=3))
            ax[1].imshow(gaussian_filter(np.exp(pmap), sigma=3))
            plt.savefig('./out/{}'.format(p))

    if which == 'imgprots':
        imgprots = buildImageProts(600)
        with open('comboimgprots.dat', 'wb') as f:
            cPickle.dump(imgprots, f, protocol=-1)

    if which == 'objprots':
        with open('comboimgprots.dat', 'rb') as f:
            imgprots = cPickle.load(f)
        objprots = buildObjProts(imgprots)
        with open('combocomplexobjprots.dat', 'wb') as f:
            cPickle.dump(objprots, f, protocol=-1)

    if which == 'run':
        targetidx = int(sys.argv[2])
        with open('comboimgprots.dat', 'rb') as f:
            imgprots = cPickle.load(f)
        with open('comboobjprots.dat', 'rb') as f:
            objprots = cPickle.load(f)
        c1out = runS1C1(sys.argv[1])
        print 'c1out done'
        s2bout = model.runS2blayer(c1out, imgprots)
        print 's2b done'
        feedback = model.feedbackSignal(objprots, targetidx)
        protidx = np.argmax(feedback)
        print 'feedback done'
        lipmap = model.topdownModulation(s2bout, feedback)
        print 'lipmap done'
        priorityMap = model.priorityMap(lipmap, [256,256])
        print priorityMap.shape

        fix, ax = plt.subplots(ncols=2)
        plt.gray()
        pmap = np.exp(np.exp(normalize(priorityMap)))
        ax[0].imshow(gaussian_filter(pmap, sigma=3))
        ax[1].imshow(gaussian_filter(np.exp(pmap), sigma=3))
        # plt.show()

        fig, ax = plt.subplots(nrows=12, ncols=3)
        plt.gray()
        i = 0
        for scale in c1out:
            cif, minV, maxV = model.imgDynamicRange(np.mean(scale, axis = 2))
            ax[i,0].imshow(cif)
            i += 1

        i = 0
        for scale in s2bout:
            #s2b, minV, maxV = model.imgDynamicRange(np.mean(scale, axis = 2))
            s2b, minV, maxV = model.imgDynamicRange(scale[:,:,protidx])
            ax[i,1].imshow(s2b)
            i += 1

        i = 0
        for scale in lipmap:
            #lm, minV, maxV = model.imgDynamicRange(np.mean(scale, axis = 2))
            lm, minV, maxV = model.imgDynamicRange(scale[:,:,protidx])  
            ax[i,2].imshow(lm)
            i += 1
        plt.show()
