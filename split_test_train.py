import os, glob
import json
import numpy as np

dataset = '/home/aby/Workspace/MTP/Datasets/Cornell/archive'

# Train/Validation Split
folders = range(1,11)
folders = ['0'+str(i) if i<10 else '10' for i in folders]
train_filenames = []
valid_filenames = []
filenames = []
count = 0
valid_img = 0
train_img = 0
# for i in folders:
#     # r.png for RGB, z.png for RGD in the end
#     for name in glob.glob(os.path.join(dataset, i, 'pcd'+i+'*z.png')):
#         if count % 5 == 0:
#             valid_filenames.append(name[len(dataset):])
#             valid_img +=1
#         else:
#             train_filenames.append(name[len(dataset):])
#             train_img +=1
#         count +=1

for i in folders:
    # r.png for RGB, z.png for RGD in the end
    for name in glob.glob(os.path.join(dataset, i, 'pcd'+i+'*z.png')):
        filenames.append(name[len(dataset):])
        count +=1

# Shuffle the list of image paths
np.random.shuffle(filenames)

# 5 Fold Cross validation (Total 885, valid: 177)
k = 5
validation_size = int(round(count / k))
for i in range(0, k):
    if i == 0:
        # 1st split as validation
        valid_filenames = filenames[0:validation_size]
        train_filenames = filenames[validation_size:]
    elif i == (k-1):
        # last split as validation
        valid_filenames = filenames[validation_size*i:]
        train_filenames = filenames[0:validation_size*i]
    else:
        # middle splits
        valid_filenames = filenames[validation_size*i:validation_size*(i+1)]
        train_filenames = filenames[0:validation_size*i]
        train_filenames.extend(filenames[validation_size*(i+1):])

    # open output file for writing
    with open(dataset + '/train_' + str(i) + '.txt', 'w') as filehandle:
        json.dump(train_filenames, filehandle)
    with open(dataset + '/valid_' + str(i) + '.txt', 'w') as filehandle:
        json.dump(valid_filenames, filehandle)
