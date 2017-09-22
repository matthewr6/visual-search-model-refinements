import json
import numpy as np
from scipy import stats

# test that first should be smaller than second
pairs = [
    # check conjunctions
    ('conjunctions', 'miconi_serial'),
    ('conjunctions', 'miconi_popout'),

    # check bw
    [],
    ('bw', 'bw_do'),
    ('bw', 'bw_so'),
    ('bw_do', 'bw_so'),

    # check colorpopout
    [],
    ('colorpopout', 'colorpopout_bw'),
    # ('colorpopout', 'colorpopout_do'),
    # ('colorpopout_bw', 'colorpopout_do'),

    # conjunctions
    [],
    ('conjunctions', 'conjunctions_do'),
    ('conjunctions', 'conjunctions_so'),
    ('conjunctions_do', 'conjunctions_so'),
    
    # check slopes against zero?
]

def zscore(m1, se1, m2, se2):
    num = m1 - m2
    denom = np.sqrt(se1**2 + se2**2)
    return num/denom

for pair in pairs:

    sizes = ['3', '6', '12', '18']
    statdata = []
    if len(pair) == 0:
        print ''
        continue
    else:
        assert len(pair) == 2
    for d in pair:
        with open('jsondata/{}.json'.format(d), 'rb') as f:
            data = json.load(f)
        points = []
        for size in sizes:
            for point in data[size]:
                points.append((int(size), point))
        unzipped = zip(*points)
        m, _, _, _, se = stats.linregress(unzipped[0], unzipped[1])
        statdata.append(m)
        statdata.append(se)
    z = zscore(*statdata)
    if z > 0: # one-tailed
        z = -z
    p = stats.norm.cdf(z)
    print pair, p