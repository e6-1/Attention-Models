import imageio
from imageio.core.util import asarray as imgToArr
import matplotlib.pylab as pylab
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import shuffle

# Bucketing parameters
num_buckets = 4

def get_bucket(num_buckets, x, y, width, height):
    delta_width = width / num_buckets
    delta_height = height / num_buckets

    for i in range(num_buckets):
        if x < i * delta_width:
            for j in range(num_buckets):
                if y < j * delta_height:
                    return (num_buckets * j + i)

# Columns: Frame, Brake, GazeX, GazeY
dataFile = './data/cleaned_data.csv'
df = pd.read_csv(dataFile, delimiter='\t')

brake = df[df['Brake'] > 0]
nonbrake = df[df['Brake'] == 0]
nonbrake = nonbrake[:len(brake)]  # Braking is far fewer than nonbraking, so trim down
df = pd.concat([brake, nonbrake])
df = df.drop(df[df['GazeX'] < 0].index)
df = df.drop(df[df['GazeY'] < 0].index)
df = df.dropna()
df = df.reset_index(drop=True)  # Resets the index to the usual 0, 1, 2, ...

filename = 'data/driving.avi'
vid = imageio.get_reader(filename,  'ffmpeg')
batch = 0
count = 0
buckets = np.zeros((1000, 16))
left_imgs = np.zeros((1000, 244, 900, 3))
braking = np.zeros((1000, 2))
img_size = (244, 900, 3)
print("Processing frames...")
indices = range(len(df))
shuffle(indices)
for index in indices:
    frame = df.loc[index, 'Frame']
    x = df.loc[index, 'GazeX']
    y = df.loc[index, 'GazeY']
    
    image = vid.get_data(frame)
    if image.shape == img_size:
        height, width, depth = image.shape

        # Store images
        imgs[count] = image
        buckets[count] = get_bucket(num_buckets, x, y, 244, 900)

        # Store gaze location
        gazes[count] = np.array([x, y])

        # increment count
        count += 1
        # Wipe and save batch
        if count % 1000 == 0:
            print("processed: {0}".format(count))
            SAVE_FILE_NAME = '/home/data2/vision6/ethpete/gaze_data/batch_{0}'.format(batch)
            np.savez_compressed(
                SAVE_FILE_NAME,
                imgs=left_imgs,
                gazes=gazes,
                buckets=buckets)
            print("Saved " + SAVE_FILE_NAME)

            # Reset
            batch += 1
            count = 0
