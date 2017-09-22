import cPickle
import sys
import numpy as np

with open(sys.argv[1], 'rb') as f:
    prots = cPickle.load(f)

print len(prots)

i = 13

print max(prots[i]['so'])
print np.mean(prots[i]['so'])

print max(prots[i]['do'])
print np.mean(prots[i]['do'])

print max(prots[i]['bw'])
print np.mean(prots[i]['bw'])