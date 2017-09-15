import cv2
import sys

luminance = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_RGB2Lab)[:,:,0]
grayscale = cv2.imread(sys.argv[1], 0)
cv2.imshow('luminance', luminance)
cv2.imshow('grayscale', grayscale)
cv2.waitKey(0)