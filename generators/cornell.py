import numpy as np
import keras
from PIL import Image
import tensorflow as tf
from skimage.transform import resize
from tensorflow.python.keras.utils.data_utils import Sequence
import math

image_sizes = [512, 640, 768, 896, 1024, 1280, 1408]

# Image Preprocessing
def central_square_crop(image, side=None):
    h, w = image.shape[0], image.shape[1]
    if side is None:
        side = np.minimum(h, w)
    begin_h = int(np.maximum(0, h - side) / 2)
    begin_w = int(np.maximum(0, w - side) / 2)
    return image[begin_h:begin_h+side, begin_w:begin_w+side, ...]

def load_and_preprocess_img(filename, side_after_crop=None, resize_height=512, resize_width=512):
    # Load Image
    image = Image.open(filename)
    # convert image to numpy array
    data = np.asarray(image)
    # Apply Central Crop
    data = central_square_crop(data, side_after_crop)
    # Convert pixel values from range (0, 255) to range (-1, 1)
    data = np.subtract(data, 127.5) # (0,255) to (-127.5,127.5)
    data = np.divide(data, 127.5) # (-1, 1)
    # Resize Image to (512, 512)
    data = resize(data, (resize_height, resize_width))
    return data

# BBox Preprocessing
def load_bboxes(name):
    '''Create a list with the coordinates of the grasping rectangles. Every 
    element is either x or y of a vertex.'''
    with open(name, 'r') as f:
        bboxes = list(map(
              lambda coordinate: float(coordinate), f.read().strip().split()))
    return bboxes

def proc_x(x, side_after_crop, init_width, width_after_resize):
    if side_after_crop < init_width:
        x = x - (init_width - side_after_crop) / 2
    x = x * width_after_resize / side_after_crop
    return x

def proc_y(y, side_after_crop, init_height, height_after_resize):
    if side_after_crop < init_height:
        y = y - (init_height - side_after_crop) / 2
    y = y * height_after_resize / side_after_crop
    return y

def bbox_to_grasp(box, side_length=512.0):
    # converting and scaling bounding boxes into grasps, g = {x, y, tan, h, w}
    x = (box[0] + (box[4] - box[0])/2)
    y = (box[1] + (box[5] - box[1])/2)
    # if box[2] == box[0]:
    #     if box[3] - box[1] > 0:
    #         # theta = math.pi/2
    #         tan_t = 1e6
    #     else:
    #         # theta = -math.pi/2
    #         tan_t = -1e6
    # else:
    #     tan_t = (box[3] -box[1]) / (box[2] -box[0])
    # tan_t = min(tan_t, 1e6)
    # tan_t = max(tan_t, -1e6)
        # theta = np.arctan( (box[3] -box[1]) / (box[2] -box[0]) )
    w = np.sqrt((box[2] -box[0]) ** 2 + (box[3] -box[1]) ** 2)
    h = np.sqrt((box[6] -box[0]) ** 2 + (box[7] -box[1]) ** 2)
    sin_t = (box[3] - box[1]) / w
    cos_t = (box[2] - box[0]) / w
    # return x, y, tan_t, h, w
    return x/side_length, y/side_length, (sin_t+1)/2, (cos_t+1)/2, h/side_length, w/side_length
    # return 2*x/side_length-1, 2*y/side_length-1, sin_t, cos_t, h/side_length, w/side_length

def grasp_to_bbox(x, y, sin_t, cos_t, h, w, side_length=512.0):
# def grasp_to_bbox(x, y, tan_t, h, w, side_length=512.0):
    # x = (x+1) * side_length/2.0
    # y = (y+1) * side_length/2.0
    # h = (h) * side_length
    # w = (w) * side_length
    # sin_t /= 10.0
    # cos_t /= 10.0
    sin_t = sin_t * 2 - 1
    cos_t = cos_t * 2 - 1
    norm_fact = (sin_t**2 + cos_t**2) ** 0.5
    sin_t = sin_t / norm_fact
    cos_t = cos_t / norm_fact
    x *= side_length
    y *= side_length
    h *= side_length
    w *= side_length
    # theta = np.arctan(tan_t)
    # sin_t = np.sin(theta)
    # cos_t = np.cos(theta)
    edge1 = (x -w/2*cos_t -h/2*sin_t, y -w/2*sin_t +h/2*cos_t)
    edge2 = (x +w/2*cos_t -h/2*sin_t, y +w/2*sin_t +h/2*cos_t)
    edge3 = (x +w/2*cos_t +h/2*sin_t, y +w/2*sin_t -h/2*cos_t)
    edge4 = (x -w/2*cos_t +h/2*sin_t, y -w/2*sin_t -h/2*cos_t)
    return [edge1, edge2, edge3, edge4]

# class CornellGenerator(keras.utils.Sequence):
class CornellGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, phi=0, batch_size=1, dim=(512,512), n_channels=3,
                 n_classes=10, shuffle=True, train=True):
        'Initialization'
        self.dim = (image_sizes[phi], image_sizes[phi])
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.train = train
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def size(self):
        """ Size of the dataset.
        """
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y_g = self.__data_generation(list_IDs_temp)

        # print('DEBUG GEN', X.shape, len(y_g), len(y_g[0]), type(y_g))
        return X, [y_g]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y_grasp = []

        # Generate data
        for i, filename in enumerate(list_IDs_temp):
            # Load Image
            image = load_and_preprocess_img(filename, resize_height=512, resize_width=512)

            # Load Bbox
            if self.train:
                bbox_file = filename[:-5]+'cpos.txt'
                bboxes = load_bboxes(bbox_file)
                # Pick 1 random bbox
                r = 8*np.random.randint(len(bboxes)/8)
                bbox = bboxes[r:r+8]
                # Modify bbox acc to img scaling
                for j in range(len(bbox)):
                    if j % 2 == 0:
                        bbox[j] = proc_x(bbox[j], side_after_crop=480, init_width=640, width_after_resize=512)
                    else:
                        bbox[j] = proc_y(bbox[j], side_after_crop=480, init_height=480, height_after_resize=512)

                # Convert to Grasp
                grasp = bbox_to_grasp(bbox)

                # Store grasp
                y_grasp.append(grasp)
            else:
                # Validation Pick all Grasps
                bbox_file = filename[:-5]+'cpos.txt'
                bboxes = load_bboxes(bbox_file)
                # Pick ALL bboxes
                y_grasp_all = []
                for r in range(int(len(bboxes)/8)):
                    q = 8*r
                    bbox = bboxes[q:q+8]
                    # Modify bbox acc to img scaling
                    for j in range(len(bbox)):
                        if j % 2 == 0:
                            bbox[j] = proc_x(bbox[j], side_after_crop=480, init_width=640, width_after_resize=512)
                        else:
                            bbox[j] = proc_y(bbox[j], side_after_crop=480, init_height=480, height_after_resize=512)

                    # Convert to Grasp
                    grasp = bbox_to_grasp(bbox)

                    # Store 1 grasp
                    y_grasp_all.append(grasp)
                # Store all grasps for an image
                y_grasp.append(y_grasp_all)

            # Store sample
            X[i,] = image
        return X, y_grasp
