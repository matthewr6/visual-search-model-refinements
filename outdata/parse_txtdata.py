import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

basename = os.path.basename(sys.argv[1]).split('.')[0]

with open(sys.argv[1], 'rb') as f:
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

for i in dict_data:
    print i
    print np.mean(dict_data[i])
    print np.std(dict_data[i])
    print ''

# let's use larger inhibition