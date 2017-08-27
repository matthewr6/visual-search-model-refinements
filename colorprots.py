import os
import random
import scipy.misc as sm
import scipy.ndimage.filters as snf
import cv2
import time
import cPickle
import numpy as np

import Model1 as model
import colorfilter as cf

def extract3DPatch(output, nbkeptweights):
    # RFsize = opt.C1RFSIZE
    RFsize = 9
    selectedScale = random.choice(range(len(output)))
    print 'Selected scale: ', selectedScale
    Cchoice =   output[selectedScale]
    print 'Cchoice shape: ', Cchoice.shape
    print 'Shape of X: ', Cchoice.shape[0], ' shape of Y: ', Cchoice.shape[1], 'and RFsize: ', RFsize
    assert Cchoice.shape[0] - RFsize > RFsize 
    shapeX = Cchoice.shape[0] - RFsize
    shapeY = Cchoice.shape[1] - RFsize
    posx = random.randrange(shapeX)
    posy = random.randrange(shapeY)
    prot = Cchoice[posx:posx+RFsize,posy:posy+RFsize,:]
    permutedweights = np.random.permutation(prot.size)
    keptweights = permutedweights[:nbkeptweights]
    zeroedweights = permutedweights[nbkeptweights:]
    prot = prot / np.linalg.norm(prot.flat[keptweights])
    prot.flat[zeroedweights] = -1
    return prot

t1 = time.time()

def buildColorProts(numProts):
    print 'Building ', numProts, 'protoypes from natural images'
    imgfiles = os.listdir('./colorrandoms')
    prots = []
    for n in range(numProts):
        selectedImg = random.choice(range(len(imgfiles)))
        print '----------------------------------------------------'
        
        imgfile = imgfiles[selectedImg]
        print 'Prot number', n, 'select image: ', selectedImg, imgfile
        
        if(imgfile == '._.DS_Store' or imgfile == '.DS_Store'):
            selectedImg = random.choice(range(len(imgfiles)))
            imgfile = imgfiles[selectedImg]
        # img = sm.imread('./colorrandoms' +'/'+imgfile)
        img = cv2.imread('./colorrandoms/' + imgfile)
        # S1outputs = runS1layer(img, s1filters)
        # C1outputs = runC1layer(S1outputs)

        out = cf.runS1C1(img)

        prots.append(extract3DPatch(out, nbkeptweights = 50))
    return prots

# print '{}s elapsed'.format(time.time() - t1)

com = ''

d = buildColorProts(600)
with open('new{}prots/colorprots.dat'.format(com), 'wb') as f:
    cPickle.dump(d, f, protocol=-1)

# this works.
# should be able to use the same s2b layer.

# color path flow - runS1C1() - s2b() - feedback - priority map

def buildObjProts(imgProts, resize=True):
    imgfiles = os.listdir('./{}colorobjs'.format(com))
    prots = [0 for i in range(len(imgfiles))]
    for n, imgfile in enumerate(imgfiles):
        tmp = imgfile.strip().split('.')
        pnum = int(tmp[0]) - 1
        print 'pnum: ', pnum

        #img = sm.imread(opt.IMAGESFOROBJPROTS+'/'+imgfile, mode='I') # changed IMAGESFOROBJPROTS to get 250 nat images c2b vals
        img = cv2.imread('./{}colorobjs/'.format(com) + imgfile)
        if resize:
            img = sm.imresize(img, (64, 64))
        t = time.time()
        out = cf.runS1C1(img)

        S2boutputs = model.runS2blayer(cf.groupnormalize(out), imgProts)
        #compute max for a given scale
        max_acts = [np.max(scale.reshape(scale.shape[0]*scale.shape[1],scale.shape[2]),axis=0) for scale in S2boutputs]
        C2boutputs = np.max(np.asarray(max_acts),axis=0) #glabal maximums
        #C2boutputs = runC1layer(S2boutputs)
        prots[pnum] = C2boutputs
        timeF = (time.time()-t)
        print "Time elapsed: ", timeF, " Estimated time of completion: ", timeF*(len(imgfiles)-(n+1))
    return prots

with open('new{}prots/colorprots.dat'.format(com), 'rb') as f:
    imgprots = cPickle.load(f)
p = buildObjProts(imgprots)
with open('new{}prots/colorobjprots.dat'.format(com), 'wb') as f:
    cPickle.dump(p, f, protocol=-1)