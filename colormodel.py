import sys
import cv2
import os
import random
import numpy as np
import cPickle
import time
import math
import matplotlib.pyplot as plt
import scipy.ndimage.filters as snf
from scipy.ndimage.filters import gaussian_filter
import Model1 as model
import scipy.misc as sm
import matplotlib.image as mpimg

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

imgdir = './images/colorrandoms'
objdir = './images/colorobjs'
def buildImageProts(numProts, single_only=False): 
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
        c1out = runS1C1(imgdir+'/'+imgfile, single_only=single_only)
        prots.append(model.extract3DPatch(c1out, nbkeptweights=300)) # 100 x (12/4) = 300
    return prots

def buildObjProts(imgProts, resize=True, single_only=False): #computing C2b
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
        img = cv2.cvtColor(cv2.imread(objdir + '/' + imgfile), cv2.COLOR_RGB2Lab)
        if resize:
            img = cv2.resize(img, (64, 64))
        
        t = time.time()
        C1outputs = runS1C1(objdir +'/'+imgfile, img, single_only=single_only)

        S2boutputs = model.runS2blayer(C1outputs, imgProts)
        #compute max for a given scale
        max_acts = [np.max(scale.reshape(scale.shape[0]*scale.shape[1],scale.shape[2]),axis=0) for scale in S2boutputs]
        C2boutputs = np.max(np.asarray(max_acts),axis=0) #glabal maximums
        #C2boutputs = runC1layer(S2boutputs)
        prots[pnum] = C2boutputs
        timeF = (time.time()-t)
        print "Time elapsed: ", timeF, " Estimated time of completion: ", timeF*(len(imgfiles)-(n+1))
    return prots

def batch_normalize(arr, highest, lowest):
    ret = []
    for m in arr:
        ret.append(normalize(m, lowest=lowest, highest=highest))
    return ret

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

def bound_array_of_mats(arr):
    m1 = 0.0
    m2 = float('inf')
    for a in arr:
        m1 = max(m1, np.max(a))
        m2 = min(m2, np.min(a))
    return (m1, m2)

def bound_dict_of_mats(d):
    m1 = 0.0
    m2 = float('inf')
    for k in d:
        a1, a2 = bound_array_of_mats(d[k])
        m1 = max(m1, a1)
        m2 = min(m2, a2)
    return (m1, m2)

so_order = ['wb','bw','rg','gr','yb','by']
# try running the shapepopout (after making prots!!! do this in separatefile in case it's bad)
# with so_flattened[i]/255.0 andsee if it makes a difference
# this would require rerunning all combo ones (colorpopout, conjunction, shapepopout) if it works
def runS1C1(imgpath, img=None, single_only=False):
    if img is None:
        img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_RGB2Lab)
    so = single_opponents(img) # six
    do = double_opponents(so) # three, also run filters on these.
    do_filtered = {}
    for d in do:
        do_filtered[d] = model.runS1layer(do[d], s1filters)
    sizes = [i.shape[:2] for i in do_filtered['wb']] # why do I have to reverse? unsure
    so_resized = {}
    for s in so:
        so_resized[s] = [cv2.resize(so[s], size[::-1]) for size in sizes] # each is 12 x n x n

    # should I normalize (separately) and how should I normalize (by orientation/color or together)

    so_flattened = transform_c1out([so_resized[s] for s in so_resized]) # ends being 12 x n x n x 6
    do_flattened = transform_c1out([do_filtered[d] for d in do_filtered]) # ends being 12 x n x n x 12

    s1out = []
    for i in range(12):
        if single_only:
            s1out.append(so_flattened[i]/255.0)
        else:
            s1out.append(np.dstack((do_flattened[i], so_flattened[i]/255.0)))
    c1out = model.runC1layer(s1out)
    return c1out

box_diam = 256/5.0
box_radius = box_diam/2.0
def check_bounds(px, py, rx, ry):
    # print (px, py, rx, ry)
    rx = (rx * box_diam) + box_radius
    ry = (ry * box_diam) + box_radius
    # print (px, py, rx, ry)
    bounds = [
        rx - box_radius,
        rx + box_radius,
        ry - box_radius,
        ry + box_radius
    ]
    return px >= bounds[0] and px <= bounds[1] and py >= bounds[2] and py <= bounds[3]

# (sidx, targetidx, target_pos[0], target_pos[1], setsize)
def parse_filename(filename):
    d = filename.split('.')[0].split('-')
    return {
        'idx': d[0],
        'targetidx': int(d[1]) - 1,
        'target_x': int(d[2]),
        'target_y': int(d[3]),
        'setsize': int(d[4])
    }

def main(which, outname, scenepath):
    if which == 'batch':
        if os.path.isfile('outdata/txtdata/{}.txt'.format(outname)):
            with open('outdata/txtdata/{}.txt'.format(outname), 'rb') as f:
                already_run = ['{}-{}'.format(a.split(' :: ')[0], a.split(' :: ')[1]) for a in f.read().split('\n') if a]
        else:
            already_run = []

        paths = os.listdir('./scenes/{}'.format(scenepath))
        with open('./prots/comboimgprots_colornorm.dat', 'rb') as f:
            imgprots = cPickle.load(f)
        with open('./prots/comboobjprots_colornorm.dat', 'rb') as f:
            objprots = cPickle.load(f)
        with open('outdata/txtdata/{}.txt'.format(outname), 'ab') as f:
            for p in paths:

                sceneinfo = parse_filename(p)

                if '{}-{}'.format(sceneinfo['idx'], sceneinfo['setsize']) in already_run:
                    continue

                print p

                c1out = runS1C1('./scenes/{}/{}'.format(scenepath, p))
                print 'c1out done'
                s2bout = model.runS2blayer(c1out, imgprots)
                print 's2b done'
                feedback = model.feedbackSignal(objprots, sceneinfo['targetidx'])
                protidx = np.argmax(feedback)
                print 'feedback done'
                lipmap = model.topdownModulation(s2bout, feedback)
                print 'lipmap done'
                priorityMap = model.priorityMap(lipmap, [256,256])
                print priorityMap.shape

                i = 0
                found = False
                fixations_allowed = sceneinfo['setsize']
                position = [sceneinfo['target_x'], sceneinfo['target_y']]
                while i < fixations_allowed and not found:
                    fx, fy = model.focus_location(priorityMap)
                    found = check_bounds(fx, fy, position[0], position[1])
                    if not found:
                        priorityMap, _, _ = model.inhibitionOfReturn(priorityMap)
                    i += 1
                print '  {}'.format(i)
                f.write('{} :: {} :: {} :: {}\n'.format(sceneinfo['idx'], sceneinfo['setsize'], i, found))
                print '{} completed'.format(p)

                # fix, ax = plt.subplots(ncols=3)
                # plt.gray()
                # pmap = np.exp(np.exp(normalize(priorityMap)))
                # ax[1].imshow(gaussian_filter(pmap, sigma=3))
                # ax[2].imshow(gaussian_filter(np.exp(pmap), sigma=3))

                # relative_focus = np.argmax(priorityMap)
                # fy = int(math.floor(relative_focus/256))
                # fx = int(relative_focus % 256)
                # img_o = mpimg.imread('./scenes/colorscenesconjunction2/{}'.format(p))
                # cv2.circle(img_o, (fx, fy), 10, (0, 0, 0))
                # ax[0].imshow(img_o)
                # plt.savefig('./outdata/outconjunction2/{}'.format(p))

    if which == 'batch_color':
        if os.path.isfile('outdata/txtdata/{}.txt'.format(outname)):
            with open('outdata/txtdata/{}.txt'.format(outname), 'rb') as f:
                already_run = ['{}-{}'.format(a.split(' :: ')[0], a.split(' :: ')[1]) for a in f.read().split('\n') if a]
        else:
            already_run = []

        paths = os.listdir('./scenes/{}'.format(scenepath))
        with open('./prots/comboimgprots_color.dat', 'rb') as f:
            imgprots = cPickle.load(f)
        with open('./prots/comboobjprots_color.dat', 'rb') as f:
            objprots = cPickle.load(f)
        with open('outdata/txtdata/{}.txt'.format(outname), 'ab') as f:
            for p in paths:

                sceneinfo = parse_filename(p)

                if '{}-{}'.format(sceneinfo['idx'], sceneinfo['setsize']) in already_run:
                    continue

                print p

                c1out = runS1C1('./scenes/{}/{}'.format(scenepath, p), single_only=True)
                print 'c1out done'
                s2bout = model.runS2blayer(c1out, imgprots)
                print 's2b done'
                feedback = model.feedbackSignal(objprots, sceneinfo['targetidx'])
                protidx = np.argmax(feedback)
                print 'feedback done'
                lipmap = model.topdownModulation(s2bout, feedback)
                print 'lipmap done'
                priorityMap = model.priorityMap(lipmap, [256,256])
                print priorityMap.shape

                i = 0
                found = False
                fixations_allowed = sceneinfo['setsize']
                position = [sceneinfo['target_x'], sceneinfo['target_y']]
                while i < fixations_allowed and not found:
                    fx, fy = model.focus_location(priorityMap)
                    found = check_bounds(fx, fy, position[0], position[1])
                    if not found:
                        priorityMap, _, _ = model.inhibitionOfReturn(priorityMap)
                    i += 1
                print '  {}'.format(i)
                f.write('{} :: {} :: {} :: {}\n'.format(sceneinfo['idx'], sceneinfo['setsize'], i, found))
                print '{} completed'.format(p)

    # so now we have L, a, b - now just run S1 and C1 from the model on them.
    if which == 'show':
        img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_RGB2Lab)
        import time
        # for i in range(3):
        #     cv2.imshow(str(i), img[:,:,i])
        # cv2.waitKey(0)
        # t = time.time()
        # runS1C1(sys.argv[1])
        # print time.time() - t
        # so = single_opponents(img)
        # for d in so:
        #     cv2.imshow(d, so[d])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # do = double_opponents(so)
        # for d in do:
        #     cv2.imshow(d, do[d])
        # cv2.waitKey(0)

    if which == 'prots_bw':
        with open('prots/imgprots.dat', 'rb') as f:
            imgprots = cPickle.load(f)
        objprots = model.buildObjProts(s1filters, imgprots, resize=True, full=True)
        with open('prots/comboobjprots_bw.dat', 'wb') as f:
            cPickle.dump(objprots, f, protocol=-1)

    if which == 'batch_bw':
        if os.path.isfile('outdata/txtdata/{}.txt'.format(outname)):
            with open('outdata/txtdata/{}.txt'.format(outname), 'rb') as f:
                already_run = ['{}-{}'.format(a.split(' :: ')[0], a.split(' :: ')[1]) for a in f.read().split('\n') if a]
        else:
            already_run = []
        paths = os.listdir('./scenes/{}'.format(scenepath))
        with open('prots/imgprots.dat', 'rb') as f:
            imgprots = cPickle.load(f)
        with open('./prots/comboobjprots_bw.dat', 'rb') as f:
            objprots = cPickle.load(f)
        with open('outdata/txtdata/{}.txt'.format(outname), 'ab') as f:
            for p in paths:
                sceneinfo = parse_filename(p)

                if '{}-{}'.format(sceneinfo['idx'], sceneinfo['setsize']) in already_run:
                    continue

                targetidx = sceneinfo['targetidx']
                img = sm.imread('./scenes/{}/{}'.format(scenepath, p), mode='I')
                S1outputs = model.runS1layer(img, s1filters)
                C1outputs = model.runC1layer(S1outputs)
                S2boutputs = model.runS2blayer(C1outputs, imgprots)
                feedback = model.feedbackSignal(objprots, targetidx)
                lipmap = model.topdownModulation(S2boutputs,feedback)

                priorityMap = model.priorityMap(lipmap,[256,256])

                i = 0
                found = False
                fixations_allowed = sceneinfo['setsize']
                position = [sceneinfo['target_x'], sceneinfo['target_y']]
                while i < fixations_allowed and not found:
                    fx, fy = model.focus_location(priorityMap)
                    found = check_bounds(fx, fy, position[0], position[1])
                    if not found:
                        priorityMap, _, _ = model.inhibitionOfReturn(priorityMap)
                    i += 1
                print '  {}'.format(i)
                f.write('{} :: {} :: {} :: {}\n'.format(sceneinfo['idx'], sceneinfo['setsize'], i, found))
                print '{} completed'.format(p)

    if which == 'imgprots':
        imgprots = buildImageProts(600)
        with open('comboimgprots.dat', 'wb') as f:
            cPickle.dump(imgprots, f, protocol=-1)

    if which == 'objprots':
        with open('prots/comboimgprots.dat', 'rb') as f:
            imgprots = cPickle.load(f)
        objprots = buildObjProts(imgprots)
        with open('prots/comboobjprots.dat', 'wb') as f:
            cPickle.dump(objprots, f, protocol=-1)

    if which == 'allprots':
        imgprots = buildImageProts(600)
        with open('prots/comboimgprots_colornorm.dat', 'wb') as f:
            cPickle.dump(imgprots, f, protocol=-1)
        with open('prots/comboimgprots_colornorm.dat', 'rb') as f:
            imgprots = cPickle.load(f)
        objprots = buildObjProts(imgprots)
        with open('prots/comboobjprots_colornorm.dat', 'wb') as f:
            cPickle.dump(objprots, f, protocol=-1)

    if which == 'prots_color':
        imgprots = buildImageProts(600, single_only=True)
        with open('prots/comboimgprots_color.dat', 'wb') as f:
            cPickle.dump(imgprots, f, protocol=-1)
        with open('prots/comboimgprots_color.dat', 'rb') as f:
            imgprots = cPickle.load(f)
        objprots = buildObjProts(imgprots, single_only=True)
        with open('prots/comboobjprots_color.dat', 'wb') as f:
            cPickle.dump(objprots, f, protocol=-1)

    if which == 'run':
        targetidx = int(sys.argv[2])
        with open('./prots/comboimgprots.dat', 'rb') as f:
            imgprots = cPickle.load(f)
        with open('./prots/comboobjprots.dat', 'rb') as f:
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

        with open('sample_priomap.dat', 'wb') as f:
            cPickle.dump(priorityMap, f, protocol=-1)

        # fix, ax = plt.subplots(ncols=2)
        # plt.gray()
        # pmap = np.exp(np.exp(normalize(priorityMap)))
        # ax[0].imshow(gaussian_filter(pmap, sigma=3))
        # ax[1].imshow(gaussian_filter(np.exp(pmap), sigma=3))
        # # plt.show()

        # fig, ax = plt.subplots(nrows=12, ncols=3)
        # plt.gray()
        # i = 0
        # for scale in c1out:
        #     cif, minV, maxV = model.imgDynamicRange(np.mean(scale, axis = 2))
        #     ax[i,0].imshow(cif)
        #     i += 1

        # i = 0
        # for scale in s2bout:
        #     #s2b, minV, maxV = model.imgDynamicRange(np.mean(scale, axis = 2))
        #     s2b, minV, maxV = model.imgDynamicRange(scale[:,:,protidx])
        #     ax[i,1].imshow(s2b)
        #     i += 1

        # i = 0
        # for scale in lipmap:
        #     #lm, minV, maxV = model.imgDynamicRange(np.mean(scale, axis = 2))
        #     lm, minV, maxV = model.imgDynamicRange(scale[:,:,protidx])  
        #     ax[i,2].imshow(lm)
        #     i += 1
        # plt.show()


# run batch_bw vs batch on color popout - currently running.

# run batch_color vs batch on shape popout to show it's notjust color deciding - second two tabs

# done:
# want to try and batch_color on conjunction also (batch_bw on conjunction is done already.)

if __name__ == '__main__':
    modes = ['run', 'show', 'imgprots', 'objprots', 'batch', 'allprots', 'batch_bw', 'prots_bw', 'prots_color', 'batch_color']

    which = 'batch'
    outname = 'shapepopout_colornorm'
    scenepath = 'shapepopout'
    
    assert which in modes
    print which, outname, scenepath

    main(which, outname, scenepath)