import sys
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# sns.set(style="white")

labels = ['Serial', 'Color-shape conjunction', 'Shape popout', 'Conjunction (SO)', 'Conjunction (DO)']
datatypes = ['miconi_serial', 'conjunction', 'miconi_popout', 'conjunction_bw', 'conjunction_color']
lines = []

for datatype in datatypes:

    sizes = ['3', '6', '12', '18']
    linebounds = [int(a) for a in sizes]

    with open('jsondata/{}.json'.format(datatype), 'rb') as f:
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
legend = plt.legend(lines, labels, loc=2)
plt.xlim(left=3, right=18)
plt.ylim(bottom=0, top=14)
plt.xticks([3,6,12,18])
plt.xlabel('Set size')
plt.ylabel('Fixation count')
plt.savefig('graphs/regressioncompare.png')
plt.clf()

labels = ['shapepopout', 'shapepopout_color', 'colorpopout', 'colorpopout_bw', 'shapepopout_doubleonly']
datatypes = ['shapepopout', 'shapepopout_color', 'colorpopout', 'colorpopout_bw', 'shapepopout_doubleonly']
lines = []

for datatype in datatypes:

    sizes = ['3', '6', '12', '18']
    linebounds = [int(a) for a in sizes]

    with open('jsondata/{}.json'.format(datatype), 'rb') as f:
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
legend = plt.legend(lines, labels, loc=2)
plt.xlim(left=3, right=18)
plt.ylim(bottom=0, top=14)
plt.xticks([3,6,12,18])
plt.xlabel('Set size')
plt.ylabel('Fixation count')
plt.savefig('graphs/regressioncompare_others.png')
plt.clf()