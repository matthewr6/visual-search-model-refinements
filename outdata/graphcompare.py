import sys
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# sns.set(style="white")

global_datatypes = [
    ['miconi_serial', 'miconi_popout', 'conjunctions'],
    ['bw', 'bw_so', 'bw_do'],
    ['colorpopout', 'colorpopout_bw', 'colorpopout_do'],
    ['colorpopout', 'bw', 'miconi_popout'],
    ['conjunctions', 'conjunctions_bw', 'conjunctions_so','conjunctions_do'],
    # ['multiconjunction', 'multiconjunction_bw', 'multiconjunction_so','multiconjunction_do'],
    # ['conjunctions', 'conjunctions_noscale', 'bw', 'bw_noscale', 'colorpopout', 'colorpopout_noscale'],
    # ['conjunctions', 'conjunctions_nofscale', 'bw', 'bw_nofscale', 'colorpopout', 'colorpopout_nofscale'],
    # ['intensityanddoubles/conjunctions', 'intensityanddoubles/conjunctions_bw', 'intensityanddoubles/conjunctions_so','intensityanddoubles/conjunctions_do'],
]
titles = ['regressioncompare', 'bw', 'colorpopout', 'originalvsnew', 'conjunctions']#,'noscale','nofscale']
captions = [
    ['Miconi Serial', 'Miconi Popout', 'Conjunctions'],
    ['Full', 'Single-opponent', 'Double-opponent'],
    ['Full', 'Black-and-white', 'Double-opponent'],
    ['Color Popout', 'Shape Popout', 'Miconi Popout'],
    ['Conjunctions', 'Black-and-white', 'Single-opponent', 'Double-opponent'],
    # ['multiconjunction', 'multiconjunction_bw', 'multiconjunction_so','multiconjunction_do'],
    # ['Conjunctions', 'Conjunctions (no feature scaling)', 'Shape Popout', 'Shape Popout (no feature scaling)', 'Color Popout', 'Color Popout (no feature scaling)'],
    # ['Conjunctions', 'Conjunctions (no exp. scaling)', 'Shape Popout', 'Shape Popout (no exp. scaling)', 'Color Popout', 'Color Popout (no exp. scaling)'],
    # ['Conjunctions', 'Black-and-white', 'Single-opponent', 'Double-opponent'],
]
assert len(titles) == len(global_datatypes)
assert len(captions) == len(titles)
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
        l = plt.plot(linebounds, m*np.array(linebounds) + b, linewidth=2)#, linestyle=lstyles[datatype])
        lines.append(l[0])

    plt.suptitle('Model Performance Regression')
    legend = plt.legend(lines, captions[i], loc=2)
    plt.xlim(left=3, right=18)
    plt.ylim(bottom=0, top=14)
    plt.xticks([3,6,12,18])
    plt.xlabel('Set size')
    plt.ylabel('Fixation count')
    plt.savefig('graphs/intensityanddoubles/{}.png'.format(titles[i]))
    plt.close()