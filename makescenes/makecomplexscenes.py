import random
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np

pathformat = './complexcolorobjs/{}.png'
# choices = np.array(range(12)) + 1
# choices = np.array(range(6)) + 6 + 1
choices = np.array(range(9)) + 1

numscenes = 25

for sidx in range(numscenes):
    np.random.shuffle(choices)
    new_im = Image.new('RGB', (256,256), '#777777')
    for i in range(3):
        for j in range(3):
            thisidx = (i*3) + j
            im = Image.open(pathformat.format(choices[thisidx]))
            im.thumbnail((65, 65))
            # if rotate:
            #     im = im.rotate(random.randint(0, 1)* 90)
            new_im.paste(im,
                (
                    (i * 90) + 6,
                    (j * 90) + 6
                )
            )

    new_im.save('./colorscenes/{}.png'.format(sidx))
    # new_im.show()