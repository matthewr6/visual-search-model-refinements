import random
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np

path = './colorobjs/old'

# print listdir(path)

for img in listdir(path):
    print img
    new_im = Image.new('RGB', (256,256), '#777777')
    old_im = Image.open(path + '/' + img)
    old_im = old_im.resize((int(256*0.75), int(256*0.75)), Image.ANTIALIAS)
    new_im.paste(old_im,
        (
            int(256 * 0.25 * 0.5),
            int(256 * 0.25 * 0.5)
        )
    )

    new_im.save('./colorobjs/{}'.format(img))
    # new_im.show()