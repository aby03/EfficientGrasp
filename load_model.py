import tensorflow as tf
from model import build_EfficientGrasp
import json
import numpy as np
from generators.cornell import load_and_preprocess_img, proc_x, proc_y, load_bboxes, bbox_to_grasp

def allow_gpu_growth_memory():
    """
        Set allow growth GPU memory to true

    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    _ = tf.Session(config = config)

# # Set before running
# allow_gpu_growth_memory()

# # Build Model
# model, prediction_model, all_layers = build_EfficientGrasp(0,
#                                                         num_classes = 10,
#                                                         num_anchors = 1,
#                                                         freeze_bn = not False,
#                                                         score_threshold = 0.5,
#                                                         print_architecture=False)

# # load pretrained weights
# model.load_weights('checkpoints/25_02_2021_16_17_24/phi_0_cornell_best_val_grasp_loss.h5', by_name=True)
# print("Done!")

## TEST ON SINGLE IMAGE
filename = '/home/aby/Workspace/MTP/Datasets/Cornell/archive/06/pcd0600r.png'

## Run on train set
dataset = '/home/aby/Workspace/MTP/Datasets/Cornell/archive'
with open(dataset+'/train.txt', 'r') as filehandle:
    train_data = json.load(filehandle)

# for i, filename in enumerate(train_data):
#     test_data = load_and_preprocess_img(filename, side_after_crop=None, resize_height=512, resize_width=512)
#     test_data = np.array(test_data)
#     test_data = test_data[np.newaxis, ...]
    
#     test_out = model.predict(test_data)
#     print('## Grasp ', i, " ##: ", test_out)
#     # print(test_out)
#     # print(type(test_out[0]))
#     # print(test_out.shape)
import math
y_grasp_all = []
# Train Print all Grasps
max_grasps = 0
for i, filename in enumerate(train_data):
    bbox_file = filename[:-5]+'cpos.txt'
    bboxes = load_bboxes(bbox_file)
    # Pick ALL bboxes
    for r in range(int(len(bboxes)/8)):
        q = 8*r
        bbox = bboxes[q:q+8]
        # Modify bbox acc to img scaling
        for j in range(len(bbox)):
            if j % 2 == 0:
                bbox[j] = proc_x(bbox[j], side_after_crop=240, init_width=640, width_after_resize=512)
            else:
                bbox[j] = proc_y(bbox[j], side_after_crop=240, init_height=480, height_after_resize=512)

        # Convert to Grasp
        grasp = bbox_to_grasp(bbox)
        if math.isnan(grasp[0]):
            print("Ghapla", i, " ", j)
            print(filename)
        # Store 1 grasp
        # print("# Grasp Label ", i, " : ", r, " #", grasp)
        y_grasp_all.append(grasp)
    # Store all grasps for an image
    # y_grasp.append(y_grasp_all)
# print(max_grasps)
grasp_np = np.asarray(y_grasp_all)
print(grasp_np.shape)
# print(np.argwhere(np.isnan(grasp_np)))
print(np.mean(grasp_np, axis = 0))
print(np.std(grasp_np, axis = 0))
exit()

# weights_list = []
# for layer in model.layers:
#     weights_list.append( layer.get_weights() ) # list of numpy arrays
# # weights = model.layers[0].get_weights()

print('done')