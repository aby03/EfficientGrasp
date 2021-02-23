import numpy as np
# import keras
from PIL import Image
# import tensorflow as tf
from skimage.transform import resize
# from tensorflow.python.keras.utils.data_utils import Sequence
import math
import matplotlib.pyplot as plt


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
    if box[2] == box[0]:
        if box[3] - box[1] > 0:
            # theta = math.pi/2
            tan_t = 1e6
        else:
            # theta = -math.pi/2
            tan_t = -1e6
    else:
        tan_t = (box[3] -box[1]) / (box[2] -box[0])
    tan_t = min(tan_t, 1e6)
    tan_t = max(tan_t, -1e6)
        # theta = np.arctan( (box[3] -box[1]) / (box[2] -box[0]) )
    w = np.sqrt((box[2] -box[0]) ** 2 + (box[3] -box[1]) ** 2)
    h = np.sqrt((box[6] -box[0]) ** 2 + (box[7] -box[1]) ** 2)
    # sin_t = (box[3] - box[1]) / w
    # cos_t = (box[2] - box[0]) / w
    return x, y, tan_t, h, w
    # return x, y, 10.0*sin_t, 10.0*cos_t, h, w
    # return 2*x/side_length-1, 2*y/side_length-1, sin_t, cos_t, h/side_length, w/side_length

# def grasp_to_bbox(x, y, sin_t, cos_t, h, w, side_length=512.0):
def grasp_to_bbox(x, y, tan_t, h, w, side_length=512.0):
    # x = (x+1) * side_length/2.0
    # y = (y+1) * side_length/2.0
    # h = (h) * side_length
    # w = (w) * side_length
    # sin_t /= 10.0
    # cos_t /= 10.0
    # norm_fact = sin_t**2 + cos_t**2
    # sin_t = sin_t / norm_fact
    # cos_t = cos_t / norm_fact
    theta = np.arctan(tan_t)
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    edge1 = (x -w/2*cos_t -h/2*sin_t, y -w/2*sin_t +h/2*cos_t)
    edge2 = (x +w/2*cos_t -h/2*sin_t, y +w/2*sin_t +h/2*cos_t)
    edge3 = (x +w/2*cos_t +h/2*sin_t, y +w/2*sin_t -h/2*cos_t)
    edge4 = (x -w/2*cos_t +h/2*sin_t, y -w/2*sin_t -h/2*cos_t)
    return [edge1, edge2, edge3, edge4]


filename = '/home/aby/Workspace/MTP/Datasets/Cornell/archive/06/pcd0600r.png'
t2_box = '/home/aby/Workspace/MTP/Datasets/Cornell/archive/06/pcd0600cpos.txt'
# Load Image
data = load_and_preprocess_img(filename)
# # To show image
data = np.add(data, 1)
data = np.multiply(data, 0.5)
plt.figure()
plt.imshow(data)
# plt.show()

# Load Bbox
bbox_file = filename[:-5]+'cpos.txt'
bboxes = load_bboxes(bbox_file)
# Pick 1 random bbox
r = 8*np.random.randint(len(bboxes)/8)
bbox = bboxes[r:r+8]
# Convert bbox acc to image transformation
for j in range(len(bbox)):
    if j % 2 == 0:
        bbox[j] = proc_x(bbox[j], side_after_crop=480, init_width=640, width_after_resize=512)
    else:
        bbox[j] = proc_y(bbox[j], side_after_crop=480, init_height=480, height_after_resize=512)
# Convert to Grasp
grasp = bbox_to_grasp(bbox)
bbox_g=grasp_to_bbox(*grasp)
# Plot bbox
bbox_np = []
for i in range(len(bbox_g)):
    bbox_np.append(bbox_g[i][0])
    bbox_np.append(bbox_g[i][1])
print(bbox_np)
# for k in range(0,9,8):
k=0
plt.plot([bbox_np[k+0],bbox_np[k+2]], [bbox_np[k+1],bbox_np[k+3]], c='red')
plt.plot([bbox_np[k+2],bbox_np[k+4]], [bbox_np[k+3],bbox_np[k+5]], c='blue')
plt.plot([bbox_np[k+4],bbox_np[k+6]], [bbox_np[k+5],bbox_np[k+7]], c='red')
plt.plot([bbox_np[k+6],bbox_np[k+0]], [bbox_np[k+7],bbox_np[k+1]], c='blue')
plt.show()
# # TEST: Convert to bbox
# bbox = grasp_to_bbox(grasp[0], grasp[1], grasp[2], grasp[3], grasp[4])


# # Load Class labels
# label = int(filename[-9:-7])
# print('Label: ', label)