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


for index in range(126):
    data = np.load('/home/data2/vision6/ethpete/gaze_data/batch_{0}.npz'.format(index))
    buckets = data['buckets']
    counts = np.bincount(buckets)
    print(counts)
