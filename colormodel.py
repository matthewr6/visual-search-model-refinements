import sys
import cv2
import os
import random
import numpy as np
import cPickle
import time
import math
import matplotlib.pyplot as plt
# import scipy.ndimage.filters as snf
from scipy.ndimage.filters import gaussian_filter
import Model1 as model
# import scipy.misc as sm
# import matplotlib.image as mpimg

s1filters = model.buildS1filters()

def runS2blayer(C1outputs, prots):
    final = {}
    for k in C1outputs:
        final[k] = model.runS2blayer(C1outputs[k], prots[k])
    return final

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

imgdir = './images/colorrandoms'
objdir = './images/colorobjs'
def buildImageProts(numProts, single_only=False, double_only=False): 
    print 'Building ', numProts, 'protoypes from natural images'
    imgfiles = os.listdir(imgdir)
    prots = {
        'bw': [],
        'do': [],
        'so': [],
    }
    for n in range(numProts):
        selectedImg = random.choice(range(len(imgfiles)))
        print '----------------------------------------------------'
        
        imgfile = imgfiles[selectedImg]
        print 'Prot number', n, 'select image: ', selectedImg, imgfile
        
        if(imgfile == '._.DS_Store' or imgfile == '.DS_Store'):
            selectedImg = random.choice(range(len(imgfiles)))
            imgfile = imgfiles[selectedImg]
        # the way I'm doing it is inefficient but works for our purposes.
        c1out = runS1C1(imgdir+'/'+imgfile, single_only=single_only, double_only=double_only)
        for p in c1out:
            prots[p].append(model.extract3DPatch(c1out[p], nbkeptweights=100))
    return prots

def buildObjProts(imgProts, resize=True, single_only=False, double_only=False): #computing C2b
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
        img_2 = cv2.imread(objdir + '/' + imgfile, 0)
        if resize:
            img = cv2.resize(img, (64, 64))
            img_2 = cv2.resize(img_2, (64, 64))
        
        t = time.time()
        C1outputs = runS1C1(objdir +'/'+imgfile, img, img_2=img_2, single_only=single_only, double_only=double_only)

        S2boutputs = runS2blayer(C1outputs, imgProts) # todo I'm right here.
        #compute max for a given scale
        maxes = {}
        for k in S2boutputs: #s2boutputs is 12 x 3(dict) xn x n x 600
            maxes[k] = [np.max(scale.reshape(scale.shape[0]*scale.shape[1],scale.shape[2]),axis=0) for scale in S2boutputs[k]]
            maxes[k] = np.max(np.asarray(maxes[k]),axis=0)
        prots[pnum] = maxes
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

def half_rectify(ch):
    ch = ch.astype(np.int16)
    o_max = np.max(ch)
    ch = ch - 128
    ch[ch < 0] = 0
    ch = np.square(ch)
    ch = normalize(ch, lowest=0.0, highest=o_max)
    return ch

def single_opponents(cielab_img):
    so = {}
    order = ['rg', 'yb'] # 'bw' for v1
    for i, _ in enumerate(order): # 90% sure v2 used the wrong indices
        so[order[i]] = cielab_img[:,:,i+1]
        so[order[i][::-1]] = 255 - cielab_img[:,:,i+1] # switch for v2/v3
        # so[order[i]] = half_rectify(cielab_img[:,:,i+1])
        # so[order[i][::-1]] = half_rectify(255 - cielab_img[:,:,i+1])
    so['in'] = cielab_img[:,:,0] # in single_opponents because it's part of the definition of color
    return so

def double_opponents(so):#, img):
    do = {}
    order = ['rg', 'yb'] # 'bw' for v1
    for o in order:
        # do[o] = cv2.absdiff(so[o] , 255 - so[o])
        do[o] = so[o] # black w/ white line is same as white w/black line so choice rg vs gr is arbitrary - this fulfills center vs surround
    # do['bw'] = img
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

def transform_c2b(c2b):
    ret = {}
    for k in c2b[0].keys():
        ap = []
        for i in c2b:
            ap.append(i[k])
        ret[k] = ap
    return ret


# expects to be in [n, 1], 0 <= n < 1
def power_transform(signal, power, domain):
    signal /= float(domain)
    signal = signal ** float(power)
    return signal
def exponential_transform(signal, domain):
    signal *= domain
    signal = (np.exp(signal) - 1.0)/(np.exp(domain) - 1.0)
    return signal
def s_curve_transform(signal):
    signal = 1.0/(1+np.exp(-10.0 * (signal - 0.5)))
    return signal


def feedbackSignals(objprots, targetIndx, feedback_scaling=True):
    feedback = {}
    fmeans = {}
    c2b = transform_c2b(objprots)
    for k in objprots[0].keys():
        C2bavg = np.mean(c2b[k],axis=0)
        C2bavg[C2bavg == 0] = float('inf')
        feedback[k] = c2b[k][targetIndx]/C2bavg

    # #v2
    # max_feedback, min_feedback = bound_dict_of_mats(feedback)
    # for k in feedback:
    #     feedback[k] = feedback[k] - min_feedback
    #     feedback[k] = feedback[k] / max_feedback
    #     # feedback[k] += 1.0

    # working on v4 with different feedback function
    # _, min_feedback = bound_dict_of_mats(feedback)
    for k in feedback:
        # let's do it individually since feature binding happens later not at this stage
        fmeans[k] = np.mean(feedback[k])
        print fmeans[k], k
        feedback[k] = normalize(feedback[k])
        if feedback_scaling:
            feedback[k] = exponential_transform(feedback[k], 4)

    return feedback, fmeans

def topdownModulation(s2boutputs, feedback):
    mapdict = {}
    for k in s2boutputs:
        assert k in feedback
        mapdict[k] = model.topdownModulation(s2boutputs[k], feedback[k], norm=k != 'so')
    return mapdict

def comboPriorityMap(lipmaps, osize, feedback_means, scaling=True):
    final = np.zeros(osize) + 1
    individuals = {}
    for k in lipmaps:
        if len(lipmaps[k]):
            newp = normalize(model.priorityMap(lipmaps[k], osize))
            if scaling:
                newp = newp * feedback_means[k]
            individuals[k] = newp
            final += newp
    final = normalize(final)
    return final, individuals

def runS1C1(imgpath, img=None, single_only=False, double_only=False, bw_only=False, img_2=None):
    print single_only, double_only
    if img is None:
        img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_RGB2Lab)
    if img_2 is None:
        do_bw = cv2.imread(imgpath, 0)
    else:
        do_bw = img_2
    so = single_opponents(img) # six
    do = double_opponents(so)#, do_bw) # three, also run filters on these so 3x4=12
    do_filtered = {}
    for d in do:
        do_filtered[d] = model.runS1layer(do[d], s1filters)

    bw = model.runS1layer(do_bw, s1filters)

    if not double_only:
        sizes = [i.shape[:2] for i in bw]
        so_resized = {}
        for s in so:
            so_resized[s] = [cv2.resize(so[s], size[::-1]) for size in sizes] # each is 12 x n x n
        so_flattened = transform_c1out([so_resized[s] for s in so_resized]) # ends being 12 x n x n x 4

    do_flattened = transform_c1out([do_filtered[d] for d in do_filtered]) # ends being 12 x n x n x 12

    s1out = {
        'bw':[],
        'so':[],
        'do':[]
    }
    #we divide by 255.0 to normalize to (1.5, -1.5)
    def so_parse(s):
        s = (s/127.5) - 1.0
        return s * 1.5
    for i in range(12):
        if single_only:
            s1out['so'].append(so_parse(so_flattened[i]))
        elif double_only:
            s1out['do'].append(do_flattened[i]) # should this include bw
            # s1out['bw'].append(bw[i])
        elif bw_only:
            s1out['bw'].append(bw[i])
        else:
            s1out['so'].append(so_parse(so_flattened[i]))
            s1out['do'].append(do_flattened[i])
            s1out['bw'].append(bw[i])
    c1out = {}
    for k in s1out:
        c1out[k] = model.runC1layer(s1out[k])
    return c1out

box_diam = 256/5.0
box_radius = box_diam/2.0
def check_bounds(px, py, rx, ry):
    rx = (rx * box_diam) + box_radius
    ry = (ry * box_diam) + box_radius
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

def main(which, outname=None, scenepath=None, feature_mode='both', prio_scaling=True, feedback_scaling=True):
    if which == 'batch':
        if os.path.isfile('outdata/txtdata/{}.txt'.format(outname)):
            with open('outdata/txtdata/{}.txt'.format(outname), 'rb') as f:
                already_run = ['{}-{}'.format(a.split(' :: ')[0], a.split(' :: ')[1]) for a in f.read().split('\n') if a]
        else:
            already_run = []

        paths = os.listdir('./scenes/{}'.format(scenepath))
        with open('./prots/comboimgprots_separatefeatures_intensityanddoubles.dat', 'rb') as f:
            imgprots = cPickle.load(f)
        with open('./prots/comboobjprots_separatefeatures_intensityanddoubles.dat', 'rb') as f:
            objprots = cPickle.load(f)
        with open('outdata/txtdata/{}.txt'.format(outname), 'ab') as f:
            for p in paths:

                sceneinfo = parse_filename(p)

                if '{}-{}'.format(sceneinfo['idx'], sceneinfo['setsize']) in already_run:
                    continue

                print p

                single_only = feature_mode in ['so', 'single', 'single_only']
                double_only = feature_mode in ['do', 'double', 'double_only']
                bw_only = feature_mode in ['bw', 'bw_only']
                c1out = runS1C1('./scenes/{}/{}'.format(scenepath, p), single_only=single_only, double_only=double_only, bw_only=bw_only)
                print 'c1out done'
                s2bout = runS2blayer(c1out, imgprots)
                print 's2b done'
                feedback, fmeans = feedbackSignals(objprots, sceneinfo['targetidx'], feedback_scaling=feedback_scaling)
                print 'feedback done'
                lipmaps = topdownModulation(s2bout, feedback)
                print 'lipmap done'
                priorityMap, _ = comboPriorityMap(lipmaps, [256,256], fmeans, prio_scaling)
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
        # import time
        # for i in range(3):
        #     cv2.imshow(str(i), img[:,:,i])
        # cv2.waitKey(0)
        # t = time.time()
        # r = runS1C1(sys.argv[1])
        # for k in r:
        #     print k
        #     for item in r[k]:
        #         print np.max(item), np.min(item)
        # print time.time() - t
        so = single_opponents(img)
        for d in so:
            cv2.imshow(d, so[d])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        do = double_opponents(so)
        for d in do:
            cv2.imshow(d, do[d])
        cv2.waitKey(0)

    if which == 'imgprots':
        imgprots = buildImageProts(600)
        with open('comboimgprots.dat', 'wb') as f:
            cPickle.dump(imgprots, f, protocol=-1)

    if which == 'objprots':
        with open('prots/comboimgprots_separatefeatures_intensityanddoubles.dat', 'rb') as f:
            imgprots = cPickle.load(f)
        objprots = buildObjProts(imgprots)
        with open('prots/comboobjprots_separatefeatures_intensityanddoubles.dat', 'wb') as f:
            cPickle.dump(objprots, f, protocol=-1)

    if which == 'allprots':
        imgprots = buildImageProts(600)
        with open('prots/comboimgprots_separatefeatures_intensityanddoubles.dat', 'wb') as f:
            cPickle.dump(imgprots, f, protocol=-1)
        with open('prots/comboimgprots_separatefeatures_intensityanddoubles.dat', 'rb') as f:
            imgprots = cPickle.load(f)
        objprots = buildObjProts(imgprots)
        with open('prots/comboobjprots_separatefeatures_intensityanddoubles.dat', 'wb') as f:
            cPickle.dump(objprots, f, protocol=-1)

    if which == 'run':
        targetidx = int(sys.argv[2])
        with open('./prots/comboobjprots_separatefeatures_intensityanddoubles.dat', 'rb') as f:
            objprots = cPickle.load(f)
        feedback = feedbackSignals(objprots, targetidx)

        with open('./prots/comboimgprots_separatefeatures_intensityanddoubles.dat', 'rb') as f:
            imgprots = cPickle.load(f)
        # protidx = np.argmax(feedback)
        print 'feedback done'
        c1out = runS1C1(sys.argv[1])
        print 'c1out done'
        s2bout = runS2blayer(c1out, imgprots)
        with open('s2bout.dat', 'wb') as f:
            cPickle.dump(s2bout, f, protocol=-1)
        print 's2b done'
        lipmaps = topdownModulation(s2bout, feedback)
        print 'lipmap done'

        # with open('sample_lipmaps.dat', 'rb') as f:
        #     lipmaps = cPickle.load(f)

        priorityMap, individualPriorityMaps = comboPriorityMap(lipmaps, [256,256])
        print priorityMap.shape

        with open('sample_lipmaps.dat', 'wb') as f:
            cPickle.dump(lipmaps, f, protocol=-1)

        fix, ax = plt.subplots(ncols=2, nrows=len(individualPriorityMaps.keys()) + 1)
        plt.gray()
        pmap = np.exp(np.exp(normalize(priorityMap)))
        ax[0,0].imshow(gaussian_filter(pmap, sigma=3))
        ax[0,1].imshow(gaussian_filter(np.exp(pmap), sigma=3))
        i = 1
        for k in individualPriorityMaps:
            print k
            print np.max(individualPriorityMaps[k])
            pmap = np.exp(np.exp(normalize(individualPriorityMaps[k])))
            ax[i,0].imshow(gaussian_filter(pmap, sigma=3))
            ax[i,1].imshow(gaussian_filter(np.exp(pmap), sigma=3))
            i += 1
        # plt.show()

        fig, ax = plt.subplots(nrows=12, ncols=3)
        plt.gray()
        i = 0
        inspectwhich = 'do'
        protidx = np.argmax(feedback[inspectwhich])
        for scale in c1out[inspectwhich]:
            cif, minV, maxV = model.imgDynamicRange(np.mean(scale, axis = 2))
            ax[i,0].imshow(cif)
            i += 1

        i = 0
        for scale in s2bout[inspectwhich]:
            #s2b, minV, maxV = model.imgDynamicRange(np.mean(scale, axis = 2))
            s2b, minV, maxV = model.imgDynamicRange(scale[:,:,protidx])
            ax[i,1].imshow(s2b)
            i += 1

        i = 0
        for scale in lipmaps[inspectwhich]:
            #lm, minV, maxV = model.imgDynamicRange(np.mean(scale, axis = 2))
            lm, minV, maxV = model.imgDynamicRange(scale[:,:,protidx])  
            ax[i,2].imshow(lm)
            i += 1
        plt.show()

# adding squares as wellto set of circles

if __name__ == '__main__':
    modes = ['run', 'show', 'imgprots', 'objprots', 'batch', 'allprots']

    which = 'show'
    assert which in modes

    # outname, scenepath, feature_mode, prio_scaling, feedback_scaling = (None, None, None, None, None)
    if which == 'batch':
        scenetypeidx = int(sys.argv[1])
        modetypeidx = int(sys.argv[2])
        prio_scaling = True
        feedback_scaling = True

        # 1-3 and then 1-4
        scenepath = ['bw', 'colorpopout', 'conjunctions', 'shapepopout', 'multiconjunction'][scenetypeidx-1]
        feature_mode = ['bw', 'so', 'do', 'both'][modetypeidx-1]

        outname = 'intensityanddoubles/{}'.format(scenepath)
        if feature_mode != 'both':
            outname = '{}_{}'.format(outname, feature_mode)
        if not prio_scaling:
            outname = '{}_noscale'.format(outname)
        if not feedback_scaling:
            outname = '{}_nofscale'.format(outname)
        print outname
        main(which, outname=outname, scenepath=scenepath, feature_mode=feature_mode, prio_scaling=prio_scaling, feedback_scaling=feedback_scaling)
    else:
        main(which)

# bw (1): 1,2,3,4






# colorpopout (2): 1,2,3

# conjunctions (3): 1,2,3,4

#colorpopout_so (2 2) and bw (1 4) not done.