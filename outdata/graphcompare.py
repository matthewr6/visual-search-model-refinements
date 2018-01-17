import sys
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# sns.set(style="white")

colors = {
    'miconi_serial': '#061539', # blues
    'miconi_popout': '#7887AB',

    'conjunctions': '#2D882D', # green

    'bw': '#686868', # shades of black/gray
    'bw_so': '#0C0C0C',
    'bw_do': '#C6C5C5',

    'colorpopout': '#AA6C39', #orange-brown
    'colorpopout_bw': '#552700',
    'colorpopout_do': '#D49A6A',
}

global_datatypes = [
    ['miconi_serial', 'miconi_popout', 'conjunctions'],
    ['bw', 'bw_so', 'bw_do'],
    ['colorpopout', 'colorpopout_bw', 'colorpopout_do'],
    ['conjunctions', 'conjunctions_noscale', 'bw', 'bw_noscale', 'colorpopout', 'colorpopout_noscale'],
    # ['colorpopout', 'bw', 'miconi_popout'],
    # ['conjunctions', 'conjunctions_bw', 'conjunctions_so','conjunctions_do'],
]
titles = ['regressioncompare', 'bw', 'colorpopout', 'noscale'] #'originalvsnew', 'conjunctions'
humantitles = [
    'Serial, parallel, and conjunction searches',
    'Shape popout',
    'Color popout',
    'No feature scaling'
    # 'originalvsnew', # not used
    # 'conjunctions', # not used
]
captions = [
    ['Serial search (model by Miconi et al)', 'Parallel search (model by Miconi et al)', 'Conjunctions'],
    ['Full model', 'Single-opponent channel only', 'Double-opponent channel only'],
    ['Full model', 'Grayscale channel only', 'Double-opponent channel only'],
    ['Conjunctions', 'Conjunctions (no feature scaling)', 'Shape Popout', 'Shape Popout (no feature scaling)', 'Color Popout', 'Color Popout (no feature scaling)'],
    # ['Color Popout', 'Shape Popout', 'Miconi Popout'],
    # ['Conjunctions', 'Grayscale channel only', 'Single-opponent channel only', 'Double-opponent channel only'],
]
assert len(titles) == len(global_datatypes)
assert len(captions) == len(titles)
assert len(captions) == len(humantitles)
n = len(titles)

for i in range(n):
    datatypes = global_datatypes[i]
    lines = []

    for datatype in datatypes:

        sizes = ['3', '6', '12', '18']
        linebounds = [int(a) for a in sizes]

        with open('jsondata/intensityanddoubles/{}.json'.format(datatype), 'rb') as f:
            data = json.load(f)

        points = []
        for size in sizes:
            for point in data[size]:
                points.append((int(size), point))

        unzipped = zip(*points)
        m, b = np.polyfit(unzipped[0], unzipped[1], 1)
        color = colors.get(datatype.replace('_noscale', ''))
        linestyle = '-'
        if 'noscale' in datatype:
            linestyle = '--' # dashed
        l = plt.plot(linebounds, m*np.array(linebounds) + b, linewidth=2, color=color, linestyle=linestyle)#, linestyle=lstyles[datatype])
        lines.append(l[0])

    plt.suptitle(humantitles[i])
    legend = plt.legend(lines, captions[i], loc=2)
    plt.xlim(left=3, right=18)
    plt.ylim(bottom=0, top=14)
    plt.xticks([3,6,12,18])
    plt.xlabel('Set size')
    plt.ylabel('Fixation count')
    plt.savefig('graphs/intensityanddoubles/{}.png'.format(titles[i]))
    plt.close()