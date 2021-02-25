import os, glob
import json
import numpy as np

dataset = '/home/aby/Workspace/MTP/Datasets/Cornell/archive'

# Train/Validation Split
folders = range(1,11)
folders = ['0'+str(i) if i<10 else '10' for i in folders]
train_filenames = []
valid_filenames = []
count = 2
valid_img = 0
train_img = 0
for i in folders:
    for name in glob.glob(os.path.join(dataset, i, 'pcd'+i+'*r.png')):
        if count % 5 == 0:
            valid_filenames.append(name[len(dataset):])
            valid_img +=1
        else:
            train_filenames.append(name[len(dataset):])
            train_img +=1
        count +=1

# Shuffle the list of image paths
np.random.shuffle(train_filenames)


# open output file for writing
with open(dataset + '/train.txt', 'w') as filehandle:
    json.dump(train_filenames, filehandle)
with open(dataset + '/valid.txt', 'w') as filehandle:
    json.dump(valid_filenames, filehandle)
