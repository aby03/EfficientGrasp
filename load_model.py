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
import json
import numpy as np
from generators.cornell import load_and_preprocess_img, proc_x, proc_y, load_bboxes, bbox_to_grasp
import matplotlib.pyplot as plt
from tensorflow import keras

# Build Model
model, prediction_model, all_layers = build_EfficientGrasp(0,
                                                        num_anchors = 1,
                                                        freeze_bn = not False,
                                                        print_architecture=False)

# load pretrained weights
# model.load_weights('checkpoints/07_03_2021_06_07_18/cornell_finish.h5', by_name=True)
model.load_weights('checkpoints/08_03_2021_19_42_08/cornell_finish.h5', by_name=True)
print("Done!")

# ## TEST ON SINGLE IMAGE
# filename = '/home/aby/Workspace/MTP/Datasets/Cornell/archive/06/pcd0600r.png'

## Run on train set
dataset = '/home/aby/Workspace/MTP/Datasets/Cornell/archive'
with open(dataset+'/train.txt', 'r') as filehandle:
    train_data = json.load(filehandle)

from dataset_processing.grasp import Grasp

for i, filename in enumerate(train_data):
    # Load Image and add batch axis
    test_data = load_and_preprocess_img(dataset+filename, side_after_crop=None, resize_height=512, resize_width=512)
    test_data = np.array(test_data, dtype=np.float32)
    test_data = test_data[np.newaxis, ...]
    
    # Load label
    gtbbs = grasp.GraspRectangles.load_from_cornell_file(dataset+filename.replace("z.png", "cpos.txt"))
    gtbbs.corner_scale((512/480,512/640))
    # Run prediction
    test_out = model.predict(test_data)
    print('## Grasp ', i, " ##: ", test_out)
    test_out = test_out[0]
    # factor = 10.0
    # test_out[2] = test_out[2] / factor
    # test_out[3] = test_out[3] / factor
    # norm_fact = (test_out[2]**2 + test_out[3]**2)**0.5
    # test_out[2] = test_out[2] / norm_fact
    # test_out[3] = test_out[3] / norm_fact
    # angle = np.arctan(test_out[2]/test_out[3])
    pred_grasp = Grasp(test_out[0:2], *test_out[2:], unnorm=True)

    ## DO TESTING
    # Layer Output
    layer_name = 'regression_c'
    intermediate_layer_model = keras.Model(inputs=model.input,
                                        outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model(test_data)
    maps_64 = np.squeeze(intermediate_output.numpy())
    maps_64 = maps_64[:64*64,:]
    maps_64 = np.reshape(maps_64, (64,64,6))

    plt.subplot(2, 3, 1)
    plt.imshow(maps_64[:,:,0])
    plt.subplot(2, 3, 2)
    plt.imshow(maps_64[:,:,1])
    plt.subplot(2, 3, 3)
    plt.imshow(maps_64[:,:,2])
    plt.subplot(2, 3, 4)
    plt.imshow(maps_64[:,:,3])
    plt.subplot(2, 3, 5)
    plt.imshow(maps_64[:,:,4])
    plt.subplot(2, 3, 6)
    plt.imshow(test_data[0,:,:,:])
    plt.show()

    # print(model.get_layer('regression_c').output)

    # Plot maps

    # Plot Grasp
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(test_data[0,:,:,:])
    pred_grasp.plot(ax, 'red')
    gtbbs.plot(ax, 0.8)
    plt.show()
    # print(test_out)
    # print(type(test_out[0]))
    # print(test_out.shape)
    # exit()

# import math
# y_grasp_all = []
# # Train Print all Grasps
# max_grasps = 0
# for i, filename in enumerate(train_data):
#     bbox_file = filename[:-5]+'cpos.txt'
#     bboxes = load_bboxes(bbox_file)
#     # Pick ALL bboxes
#     for r in range(int(len(bboxes)/8)):
#         q = 8*r
#         bbox = bboxes[q:q+8]
#         # Modify bbox acc to img scaling
#         for j in range(len(bbox)):
#             if j % 2 == 0:
#                 bbox[j] = proc_x(bbox[j], side_after_crop=240, init_width=640, width_after_resize=512)
#             else:
#                 bbox[j] = proc_y(bbox[j], side_after_crop=240, init_height=480, height_after_resize=512)

#         # Convert to Grasp
#         grasp = bbox_to_grasp(bbox)
#         if math.isnan(grasp[0]):
#             print("Ghapla", i, " ", j)
#             print(filename)
#         # Store 1 grasp
#         # print("# Grasp Label ", i, " : ", r, " #", grasp)
#         y_grasp_all.append(grasp)
#     # Store all grasps for an image
#     # y_grasp.append(y_grasp_all)
# # print(max_grasps)
# grasp_np = np.asarray(y_grasp_all)
# print(grasp_np.shape)
# # print(np.argwhere(np.isnan(grasp_np)))
# print(np.mean(grasp_np, axis = 0))
# print(np.std(grasp_np, axis = 0))
# exit()

# weights_list = []
# for layer in model.layers:
#     weights_list.append( layer.get_weights() ) # list of numpy arrays
# # weights = model.layers[0].get_weights()

print('done')