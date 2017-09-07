import random
from PIL import Image
from os import listdir
from os.path import isfile, join
import numpy as np

pathformat = '../images/colorobjs/{}.png'
choices = {
    'circles': {
        'red': 1,
        'green': 5,
    },
    'rectangles': {
        'red': 7,
        'green': 11,
    }
}
color_choices = choices['circles'].keys()
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
    target_shape = random.choice(['circles', 'rectangles'])
    target_color = random.choice(color_choices)
    other_color = target_color
    while other_color == target_color:
        other_color = random.choice(color_choices)
    other_shape = 'circles' if target_shape == 'rectangles' else 'rectangles'

    target_idx = choices[target_shape][target_color]
    nottarget = choices[target_shape][other_color]
    distractoridx = choices[other_shape][target_color]

    return (target_idx, nottarget, distractoridx)

scenes_per_setsize = 100
setsizes = [3, 6, 12, 18]

for setsize in setsizes:
    for sidx in range(scenes_per_setsize):
        targetidx, distractor1, distractor2 = get_objs()
        target_pos, distractor_positions = get_positions(setsize)
        new_im = Image.new('RGB', (256,256), '#777777')
        for i in range(5):
            for j in range(5):
                im = None
                if [i, j] == target_pos:
                    im = Image.open(pathformat.format(targetidx))
                elif [i, j] in distractor_positions:
                    if random.randint(0, 1) == 1:
                        # im = Image.open(pathformat.format(targetidx)).rotate(90)
                        im = Image.open(pathformat.format(distractor1))
                    else:
                        im = Image.open(pathformat.format(distractor2))
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

        new_im.save('../scenes/redgreenconjunctions/{}-{}-{}-{}-{}.png'.format(sidx, targetidx, target_pos[0], target_pos[1], setsize))
        # new_im.show()