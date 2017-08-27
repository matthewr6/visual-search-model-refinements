import urllib
import random

num_imgs = 250 

base_url = 'http://lorempixel.com/480/320/{}'

opts = [
    'animals',
    'business',
    'city',
    'food',
    'nightlife',
    'fashion',
    'people',
    'nature',
    'sports',
    'transport',
]

for i in range(num_imgs):
    urllib.urlretrieve(base_url.format(random.choice(opts)), 'colorrandoms/{}.png'.format(i))
