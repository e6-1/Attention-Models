import imageio
from imageio.core.util import asarray as imgToArr
import matplotlib.pylab as pylab
from math import sqrt
import numpy as np
import pandas as pd
from random import shuffle
import scipy.misc

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from PIL import Image

counts = {}
for index in range(120, 126):
    data = np.load('/home/data2/vision6/ethpete/gaze_data/batch_{0}.npz'.format(index))
    buckets = data['buckets']
    for bucket in buckets:
        b = np.argmax(bucket)
        if b in counts:
            counts[b] += 1
        else:
            counts[b] = 1
print(counts)
