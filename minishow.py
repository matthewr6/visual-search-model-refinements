import sys
import cv2

def single_opponents(cielab_img):
    so = {}
    order = ['rg', 'yb'] # 'bw' for v1
    for i, _ in enumerate(order): # 90% sure v2 used the wrong indices
        so[order[i]] = cielab_img[:,:,i+1]
        so[order[i][::-1]] = 255 - cielab_img[:,:,i+1] 
    return so

def double_opponents(so, img):
    do = {}
    order = ['rg', 'yb'] # 'bw' for v1
    for o in order:
        do[o] = cv2.absdiff(so[o] , 255 - so[o])
    do['bw'] = img
    return do

def main():
    img = cv2.cvtColor(cv2.imread(sys.argv[1]), cv2.COLOR_RGB2Lab)
    so = single_opponents(img)
    for d in so:
        cv2.imshow(d, so[d])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    do = double_opponents(so, cv2.imread(sys.argv[1], 0))
    for d in do:
        cv2.imshow(d, do[d])
    cv2.waitKey(0)

# adding squares as wellto set of circles

if __name__ == '__main__':
    main()