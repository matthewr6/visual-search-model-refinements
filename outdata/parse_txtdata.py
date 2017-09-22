import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

basenames = ['bw', 'bw_do', 'bw_so', 'colorpopout', 'colorpopout_bw',
             'colorpopout_do', 'conjunctions_bw', 'conjunctions_so',
             'conjunctions_do', 'multiconjunction', 'multiconjunction_bw',
             'multiconjunction_do', 'multiconjunction_so', 'bw_noscale',
            'conjunctions_noscale', 'colorpopout_noscale', 'bw_nofscale',
            'conjunctions_nofscale', 'colorpopout_nofscale']

for basename in basenames:

    with open('txtdata/{}.txt'.format(basename), 'rb') as f:
        o_data = f.read().split('\n')[:-1]

    dict_data = {}
    list_data = []
    for row in o_data:
        row = row.split(' :: ')
        found = row[2] == 'True'
        # if not found:
        #     continue
        fixations = int(row[2])
        setsize = row[1]
        if setsize not in dict_data:
            dict_data[setsize] = []
        dict_data[setsize].append(fixations)
        list_data.append(int(row[2]))

    with open('jsondata/{}.json'.format(basename), 'wb') as f:
        json.dump(dict_data, f, indent=4)
        # json.dump(list_data, f, indent=4)

    # list_data = np.array(list_data)
    # x = np.arange(9) + 1
    # y = [len(list_data[list_data <= i]) for i in x]
    # plt.plot(x, y)
    # plt.ylim([0, 100])
    # plt.show()

    print basename
    for i in dict_data:
        print i
        print '   ', np.mean(dict_data[i])
        print '   ', np.std(dict_data[i])
    print ''

    # let's use larger inhibition