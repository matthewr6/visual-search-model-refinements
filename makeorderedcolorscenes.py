import random
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np

pathformat = './colorobjs/{}.png'
# choices = np.array(range(12)) + 1
# choices = np.array(range(6)) + 6 + 1
choices = np.array(range(6)) + 1

single_distractor = True
rotate = False
numscenes = 25

targetidx = 8 # color A, shape A
nottarget = 12 # color B, shape A
distractoridx = 2 # color A, shape B

for sidx in range(numscenes):
    new_im = Image.new('RGB', (256,256), '#777777')

    tx = random.randint(0, 2) # inclusive
    ty = random.randint(0, 2)
    # if single_distractor:
    #     distractoridx = targetidx
    #     while distractoridx == targetidx:
    #         distractoridx = random.choice(choices)
    for i in range(3):
        for j in range(3):
            if i == tx and j == ty:
                im = Image.open(pathformat.format(targetidx))
            else:
                if random.randint(0, 1) == 1:
                    # im = Image.open(pathformat.format(targetidx)).rotate(90)
                    im = Image.open(pathformat.format(nottarget))
                else:
                    im = Image.open(pathformat.format(distractoridx))
            im.thumbnail((50, 50))
            # if rotate:
            #     im = im.rotate(random.randint(0, 1)* 90)
            new_im.paste(im,
                (
                    (i * 90) + 12,
                    (j * 90) + 12
                )
            )

    new_im.save('./colorscenes/{}-{}-{}-{}.png'.format(sidx, targetidx,tx, ty))
    # new_im.show()