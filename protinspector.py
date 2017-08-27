import cPickle
import sys
import numpy as np

assert sys.argv[1][-3:] == 'dat'

with open(sys.argv[1], 'rb') as f:
    prot = cPickle.load(f)

print prot[0].shape