import json
import numpy as np
from scipy import stats

# test that first should be smaller than second
pairs = [
    ('conjunctions', 'conjunctions_singleonly'),
    ('conjunctions', 'conjunctions_doubleonly'),
    ('singlevsboth', 'singlevsboth_singleonly'),
]

def zscore(m1, se1, m2, se2):
    num = m1 - m2
    denom = np.sqrt(se1**2 + se2**2)
    return num/denom

for pair in pairs:

    sizes = ['3', '6', '12', '18']
    statdata = {}
    assert len(pair) == 2
    for d in pair:
        with open('jsondata/{}.json'.format(d), 'rb') as f:
            data = json.load(f)
        statdata[d] = data
    for d in statdata[pair[0]]:
        print stats.ttest_ind(statdata[pair[0]][d], statdata[pair[1]][d])[1]
    print pair