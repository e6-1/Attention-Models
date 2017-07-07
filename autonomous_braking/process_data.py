import imageio
from imageio.core.util import asarray as imgToArr
import matplotlib.pylab as pylab
import numpy as np
import pandas as pd

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
batch = 1
count = 1
left_imgs = np.zeros((10000, 244, 450, 3))
right_imgs = np.zeros((10000, 244, 450, 3))
center_imgs = np.zeros((10000, 244, 450, 3))
gazes = np.zeros((10000, 2))
braking = np.zeros((10000, 2))
img_size = (244, 900, 3)
for i, row in df.iterrows():
    frame = row['Frame']
    x = row['GazeX']
    y = row['GazeY']
    brake = row['Brake']
    
    image = vid.get_data(frame)
    if image.shape == img_size:
        height, width, depth = image.shape
        left_img = image[:, :width/2, :]
        right_img = image[:, width/2:, :]
        center_img = image[:, (width/2 - width/4): (width/2 + width/4), :]

        # Store images
        left_imgs[count] = left_img
        right_imgs[count] = right_img
        center_imgs[count] = center_img

        # Store gaze location
        gazes[count] = np.array([x, y])
        
        # Store braking info
        if brake == 0:
            braking[count] = np.array([1, 0])  # nonbraking on left
        else:
            braking[count] = np.array([0, 1])  # braking on right

        # increment count
        count += 1

        # Wipe and save batch
        if count % 10000 == 0:
            print("processed: {0}".format(count))
            SAVE_FILE_NAME = 'data/batch_{0}'.format(batch)
            np.savez_compressed(
                SAVE_FILE_NAME,
                left_imgs=left_imgs,
                right_imgs=right_imgs,
                center_imgs=center_imgs,
                gazes=gazes,
                braking=braking)
            print("Saved " + SAVE_FILE_NAME)

            # Reset
            batch += 1
            count = 1