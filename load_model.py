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
from generators.cornell import load_and_preprocess_img, proc_x, proc_y, load_bboxes, bbox_to_grasp
import matplotlib.pyplot as plt
from tensorflow import keras
from dataset_processing.grasp import Grasp
from losses import grasp_loss

from shapely import speedups
speedups.disable()
from shapely.geometry import Polygon # For IoU

# Build Model
model, prediction_model, all_layers = build_EfficientGrasp(0,
                                                        num_anchors = 1,
                                                        freeze_bn = not False,
                                                        print_architecture=False)

# load pretrained weights
# model.load_weights('checkpoints/07_03_2021_06_07_18/cornell_finish.h5', by_name=True)
model.load_weights('checkpoints/09_03_2021_18_44_01/cornell_finish.h5', by_name=True)
print("Done!")

# ## TEST ON SINGLE IMAGE
# filename = '/home/aby/Workspace/MTP/Datasets/Cornell/archive/06/pcd0600r.png'

## Run on train set
dataset = '/home/aby/Workspace/MTP/Datasets/Cornell/archive'
with open(dataset+'/train.txt', 'r') as filehandle:
    train_data = json.load(filehandle)

from dataset_processing.cornell_data import CornellDataset

test_generator = CornellDataset(
    dataset,
    train_data,
    train=False,
    shuffle=False,
    run_test=True,
)


for i, [X, Y] in enumerate(test_generator):
    X = np.float32(X)
    gtbbs = Y
    test_out = model.predict(X)
    print('## Grasp ', i, " ##: ", test_out)
    test_out = test_out[0]
    pred_grasp = Grasp(test_out[0:2], *test_out[2:], unnorm=True)

    ## DO TESTING
    # Metrics
    # For a labelled grasp
    for j in range(len(gtbbs.grs)):
        # Grasp Loss
        cur_loss = grasp_loss(gtbbs[j].as_grasp.as_list, test_out)

        # IoU Angle Diff
        # Converted to Grasp obj, unnormalized, in [y,x] format
        true_grasp_obj = gtbbs[j].as_grasp
        pred_grasp_obj = Grasp(test_out[0:2], *test_out[2:], unnorm=True)
        # converted to list of bboxes in [x, y] format
        bbox_true = true_grasp_obj.as_bbox
        bbox_pred = pred_grasp_obj.as_bbox
        
        #IoU
        try:
            p1 = Polygon([bbox_true[0], bbox_true[1], bbox_true[2], bbox_true[3], bbox_true[0]])
            p2 = Polygon([bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3], bbox_pred[0]])
            iou = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area)
        except Exception as e: 
            print('IoU ERROR', e)
            print('Bbox pred:', bbox_pred)
            print('pred grasp:', pred_grasps[j])
            print('Bbox true:', bbox_true)
        
        #Angle Diff
        true_sin = true_grasp_obj.sin_t
        true_cos = true_grasp_obj.cos_t  
        if true_cos != 0:
            true_angle = np.arctan(true_sin/true_cos) * 180/np.pi
        else:
            true_angle = 90
        
        pred_sin = pred_grasp_obj.sin_t
        pred_cos = pred_grasp_obj.cos_t
        if pred_cos != 0:
            pred_angle = np.arctan(pred_sin/pred_cos) * 180/np.pi
        else:
            pred_angle = 90
        
        angle_diff = np.abs(pred_angle - true_angle)
        angle_diff = min(angle_diff, 180.0 - angle_diff)

        if angle_diff < 30. and iou >= 0.25:
            print('CORE Grasp {}, Loss {}, IoU {}, Angle_diff {}'.format(j, cur_loss, iou, angle_diff))
        else:
            print('INCO Grasp {}, Loss {}, IoU {}, Angle_diff {}'.format(j, cur_loss, iou, angle_diff))
            
            # print('image: %d | duration = %.2f | count = %d | iou = %.2f | angle_difference = %.2f' %(step, duration, count, iou, angle_diff))
    # grasp_accuracy = correct_grasp_count / len(true_grasps)
    # avg_iou = sum(iou_list) / len(true_grasps)
    # avg_angle_diff = sum(angle_diff_list) / len(true_grasps)

    # Layer Output
    # layer_name = 'feature_concat'regression_c
    layer_name = 'regression_c'
    intermediate_layer_model = keras.Model(inputs=model.input,
                                        outputs=model.get_layer(layer_name).output)
    intermediate_output = intermediate_layer_model(X)
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
    plt.imshow(X[0,:,:,:])
    plt.show()

    # print(model.get_layer('regression_c').output)

    # Plot maps

    # Plot Grasp
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.imshow(X[0,:,:,:])
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