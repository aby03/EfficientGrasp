import glob
import os

#from utils.dataset_processing 
from dataset_processing import grasp, image
# from .grasp_data import GraspDatasetBase
import numpy as np
import random
from tensorflow.python.keras.utils.data_utils import Sequence

#Debug
import json

class CornellDataset(Sequence):
    """
    Dataset wrapper for the Cornell dataset.
    """

    def __init__(self, dataset_path, list_IDs, phi=0, batch_size=1, output_size=512, n_channels=3,
                 n_classes=10, shuffle=True, train=True, ds_rotate=0):
        """
        :param output_size: Image output size in pixels (square)
        :param random_rotate: Whether random rotations are applied
        :param random_zoom: Whether random zooms are applied
        :param input_only: Whether to return only the network input (no labels)
        :param dataset_path: Cornell Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        """
        self.random_rotate = train
        self.random_zoom = train

        # Generator
        self.output_size = output_size
        self.batch_size = batch_size
        self.dataset = dataset_path
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.train = train

        # List of rgd files of train/valid split
        self.rgd_files = list_IDs
        self.rgd_files.sort()
        # List of grasp files
        self.grasp_files = [f.replace('z.png', 'cpos.txt') for f in self.rgd_files]
        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(dataset_path))

        self.indexes = np.arange(len(self.rgd_files))
        self.on_epoch_end()

        ## For cross validation on object split (Instead of shuffle, rotate and pick 80% for train data)
        # if ds_rotate:
        #     self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
        #                                                                          :int(self.length * ds_rotate)]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.rgd_files) / self.batch_size))

    def size(self):
        """ Size of the dataset.
        """
        return len(self.rgd_files)

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.dataset+self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):
        gtbbs = grasp.GraspRectangles.load_from_cornell_file(self.dataset+self.grasp_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        gtbbs.rotate(rot, center)
        gtbbs.offset((-top, -left))
        gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_rgd(self, idx, rot=0, zoom=1.0, normalise=True):
        rgd_img = image.Image.from_file(self.dataset+self.rgd_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgd_img.rotate(rot, center)
        rgd_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgd_img.zoom(zoom)
        rgd_img.resize((self.output_size, self.output_size))
        if normalise:
            rgd_img.normalise()
            # rgd_img.img = rgd_img.img.transpose((2, 0, 1))
        return rgd_img.img

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y_g = self.__data_generation(indexes)

        return X, [y_g]    

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.output_size, self.output_size, self.n_channels))
        y_grasp = []

        # For every image in batch
        for i in range(indexes.shape[0]):
            if self.train:
                # If training data
                # Rotation augmentation
                if self.random_rotate:
                    rotations = [0, np.pi / 2, 2 * np.pi / 2, 3 * np.pi / 2]
                    rot = random.choice(rotations)
                else:
                    rot = 0.0
                # Zoom Augmentation
                if self.random_zoom:
                    zoom_factor = np.random.uniform(0.5, 1.0)
                else:
                    zoom_factor = 1.0
                
                # Load image with zoom and rotation
                rgd_img = self.get_rgd(indexes[i], rot, zoom_factor)

                # Load bboxes
                gtbb = self.get_gtbb(indexes[i], rot, zoom_factor)
                # Pick random grasp
                g_id = np.random.randint(len(gtbb.grs))
                # HARDCODE TO PICK ONLY 1st GRASP
                g_id = 0
                # Get Grasp as list [x y angle(in rad) h w]
                grasp = (gtbb[g_id].as_grasp).as_list 

                # Store grasp
                y_grasp.append(grasp)

                # print(gtbb.grs[0])
                # print(self.rgd_files[indexes[i]])
                # print('Bbox count: ', len(gtbb.grs))
                # print(grasp)

                # # Display all Grasps
                # import matplotlib.pyplot as plt
                # fig = plt.figure()
                # ax = fig.add_axes([0,0,1,1])
                # ax.imshow(rgd_img)
                # gtbb.plot(ax, 1)
                # plt.show()
            else:
                # If validation data
                # Load image with 1 zoom and 0 rotation
                rgd_img = self.get_rgd(indexes[i], 0, 1)

                # Load bboxes
                gtbb = self.get_gtbb(indexes[i], 0, 1)
                # Pick all grasps
                y_grasp_image = []
                for g_id in range(len(gtbb.grs)):
                    # Get Grasp as list [x y angle(in rad) h w]
                    grasp = (gtbb[g_id].as_grasp).as_list 
                    # Store each grasp for an image
                    y_grasp_image.append(grasp)

                # Store all grasps for an image
                y_grasp.append(y_grasp_image)

                # print(len(y_grasp))
                # print(len(y_grasp[0]))
                # # Display all Grasps
                # import matplotlib.pyplot as plt
                # fig = plt.figure()
                # ax = fig.add_axes([0,0,1,1])
                # ax.imshow(rgd_img)
                # gtbb.plot(ax, 1)
                # plt.show()
                
            # Store sample
            X[i,] = rgd_img
        return X, np.asarray(y_grasp)

# ### TESTING
# dataset = "/home/aby/Workspace/MTP/Datasets/Cornell/archive"
# with open(dataset+'/train.txt', 'r') as filehandle:
#     train_data = json.load(filehandle)

# train_generator = CornellDataset(
#     dataset,
#     train_data,
#     train=False,
#     shuffle=False,
#     batch_size=2
# )

# train_generator[0]