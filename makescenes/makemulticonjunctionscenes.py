import random
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np

pathformat = '../images/colorobjs/{}.png'
choices = [15, 16]
position_choices = np.dstack(np.mgrid[:5,:5]).reshape(25, 2)

def get_positions(n):
    target_position = random.choice(position_choices).tolist()
    distractor_positions = []
    for i in range(n-1):
        new_pos = target_position
        while new_pos == target_position or new_pos in distractor_positions:
            new_pos = random.choice(position_choices).tolist()
        distractor_positions.append(new_pos)
    return (target_position, distractor_positions)

def get_objs():
    target_idx = random.choice(choices)
    distractoridx = 15 if target_idx == 16 else 16

    return (target_idx, distractoridx)

scenes_per_setsize = 100
setsizes = [3, 6, 12, 18]

# 5 - green circle, 4 - pink circle, 10 - pink line, 11 - green
# targetidx = 10 # color A, shape A
# nottarget = 11 # color B, shape A
# distractoridx = 4 # color A, shape B

for setsize in setsizes:
    for sidx in range(scenes_per_setsize):
        targetidx, distractoridx = get_objs()
        target_pos, distractor_positions = get_positions(setsize)
        new_im = Image.new('RGB', (256,256), '#777777')
        for i in range(5):
            for j in range(5):
                im = None
                if [i, j] == target_pos:
                    im = Image.open(pathformat.format(targetidx))
                elif [i, j] in distractor_positions:
                    im = Image.open(pathformat.format(distractoridx))
                if im is not None:
                    im.thumbnail((30, 30))
                    # if rotate:
                    #     im = im.rotate(random.randint(0, 1)* 90)
                    new_im.paste(im,
                        (
                            (i * 52) + 8,
                            (j * 52) + 8
                        )
                    )

        new_im.save('../scenes/multiconjunction/{}-{}-{}-{}-{}.png'.format(sidx, targetidx, target_pos[0], target_pos[1], setsize))
        # new_im.show()