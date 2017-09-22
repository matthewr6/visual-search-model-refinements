S1RFSIZES = range(7,30,2)
# S1RFSIZES = range(3,26,2)
# S1RFSIZES = range(5,26,2)
NBS1SCALES = len(S1RFSIZES)
C1RFSIZE = 9


# Additive constants in the denominator for the normalizations in the S 
# stages.
SIGMAS = .5 
SIGMAS1 = 0
STRNORMLIP = 5

NBPROTS = 600
NBS3PROTS = 1720

NBKEPTWEIGHTS = 100

IMAGESFORPROTS = './naturalimages'
# IMAGESFOROBJPROTS = './objectimages'
# IMAGESFOROBJPROTS = './gdrivesets/objs'
IMAGESFOROBJPROTS = './images/colorobjs'
# IMAGESFOROBJPROTS = './complexcolorobjs'

# GAUSSFACTOR = 150.0
# IORSIGMA = 35
# IORSIGMA = 17.5
# IORSIGMA = 17.5
# GAUSSFACTOR = 25.0
# IORSIGMA = 35
# IORSIGMA = 10.0
# IORSIGMA = 12.5
IORSIGMA = 15.0
# GAUSSFACTOR = 150.0
# GAUSSFACTOR = 10.0
# GAUSSFACTOR = 10.0
GAUSSFACTOR = 500.0
# GAUSSFACTOR = 7.5

# 15 and 7.5
# try 25 for sigma?

# default is better than sigma 10 I think