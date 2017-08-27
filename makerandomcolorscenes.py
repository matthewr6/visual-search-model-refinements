import random
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np

pathformat = './colorobjs/{}.png'
choices = np.array(range(12)) + 1
# choices = np.array(range(6)) + 6 + 1
# choices = np.array(range(6)) + 1# + 6

single_distractor = False
no_target = False
colors = [2, 4]
rotate = False
numscenes = 100

for sidx in range(numscenes):
    new_im = Image.new('RGB', (256,256), '#777777')

    targetidx = random.choice(choices)

    tx = random.randint(0, 2)
    ty = random.randint(0, 2)
    if single_distractor:
        distractoridx = targetidx
        while distractoridx == targetidx:
            distractoridx = random.choice(choices)
    for i in range(3):
        for j in range(3):
            if not no_target:
                if i == tx and j == ty:
                    im = Image.open(pathformat.format(targetidx))
                else:
                    if not single_distractor:
                        distractoridx = targetidx# should I have a single or multiple
                        while distractoridx == targetidx:
                            distractoridx = random.choice(choices)
                    im = Image.open(pathformat.format(distractoridx))
            else:
                im = Image.open(pathformat.format(random.choice(colors)))
            im.thumbnail((50, 50))
            if rotate:
                im = im.rotate(random.randint(0, 1)* 90)
            new_im.paste(im,
                (
                    (i * 90) + 12,
                    (j * 90) + 12
                )
            )
    if no_target:
        if sidx < 50:
            targetidx = 2
        else:
            targetidx = 4
    new_im.save('./colorscenes/{}-{}-{}-{}.png'.format(sidx, targetidx, tx, ty))
    # new_im.show()