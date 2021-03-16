import tensorflow as tf
from dataset_processing import grasp

def allow_gpu_growth_memory():
    """
        Set allow growth GPU memory to true
    """
    # Eager execution
    tf.compat.v1.enable_eager_execution()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    _ = tf.Session(config = config)

# Set before importing and running
allow_gpu_growth_memory()

# Import rest
from model import build_EfficientGrasp
# from model_split import build_EfficientGrasp
import json
import numpy as np
from tensorflow import keras
from dataset_processing.grasp import Grasp
from dataset_processing.cornell_data import CornellDataset

import matplotlib.pyplot as plt

# Build Model
model, prediction_model, all_layers = build_EfficientGrasp(0,
                                                        num_anchors = 1,
                                                        freeze_bn = not False,
                                                        print_architecture=False)

# load pretrained weights
model.load_weights('checkpoints/10_03_2021_04_00_38/cornell_best_grasp_accuracy.h5', by_name=True)
print("Weights loaded!")


run_multiple = True
if run_multiple:
    # Load list of images
    dataset = '/home/aby/Workspace/MTP/Datasets/Cornell/archive'
    with open(dataset+'/valid.txt', 'r') as filehandle:
        train_data = json.load(filehandle)

    # Visualization on Custom Images
    for i, filename in enumerate(train_data):
        # Load Images
        X = CornellDataset.load_custom_image(dataset+filename)
        disp_img = CornellDataset.load_custom_image(dataset+filename, normalise=False)
        # Expand dim for batch
        X = X[np.newaxis, ...]
        Y_pred = model.predict(X)
        # Remove batch dim
        test_out = Y_pred[0]
        pred_grasp = Grasp(test_out[0:2], *test_out[2:], unnorm=True)

        # Plot predicted grasp
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.imshow(disp_img)
        pred_grasp.plot(ax, 'red')
        plt.show()
else:
    ## TEST ON SINGLE IMAGE
    filename = '/home/aby/Workspace/MTP/Datasets/Cornell/archive/06/pcd0600r.png'
    # Load Image
    X = CornellDataset.load_custom_image(filename)
    disp_img = CornellDataset.load_custom_image(filename, normalise=False)
    # Expand dim for batch
    X = X[np.newaxis, ...]
    Y_pred = model.predict(X)
    # Remove batch dim
    test_out = Y_pred[0]
    pred_grasp = Grasp(test_out[0:2], *test_out[2:], unnorm=True)

    # Plot predicted grasp
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(disp_img)
    pred_grasp.plot(ax, 'red')
    plt.show()
