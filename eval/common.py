"""
EfficientPose (c) by Steinbeis GmbH & Co. KG für Technologietransfer
Haus der Wirtschaft, Willi-Bleicher-Straße 19, 70174 Stuttgart, Germany
Yannick Bukschat: yannick.bukschat@stw.de
Marcus Vetter: marcus.vetter@stw.de

EfficientPose is licensed under a
Creative Commons Attribution-NonCommercial 4.0 International License.

The license can be found in the LICENSE file in the root directory of this source tree
or at http://creativecommons.org/licenses/by-nc/4.0/.
---------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------

Based on:

Keras EfficientDet implementation (https://github.com/xuannianz/EfficientDet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
The official EfficientDet implementation (https://github.com/google/automl) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
EfficientNet Keras implementation (https://github.com/qubvel/efficientnet) licensed under the Apache License, Version 2.0
---------------------------------------------------------------------------------------------------------------------------------
Keras RetinaNet implementation (https://github.com/fizyr/keras-retinanet) licensed under
    
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# from utils.compute_overlap import compute_overlap, wrapper_c_min_distances
from utils.visualization import draw_detections, draw_annotations

import tensorflow as tf
import numpy as np
import os
import math
from tqdm import tqdm

import cv2
import progressbar

from losses import grasp_loss

assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator, model, save_path = None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = (boxes+classes = detections[num_detections, 4 + num_classes], rotations = detections[num_detections, num_rotation_parameters], translations = detections[num_detections, num_translation_parameters)

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """

    # all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]

    pred_grasps = [None for i in range(generator.size()) ]
    true_grasps = [None for i in range(generator.size()) ]

    for i in progressbar.progressbar(range(generator.size()), prefix='Running network: '):
        image_bt, output_bt    = generator[i]
        true_grasp_bt = output_bt[0]
        # raw_image    = generator.load_image(i)
        # image, scale = generator.preprocess_image(raw_image.copy())
        # image, scale = generator.resize_image(image)

        # if keras.backend.image_data_format() == 'channels_first':
        #     image = image.transpose((2, 0, 1))

        # run network
        pred_grasp_bt = model.predict_on_batch(image_bt)

        pred_grasps[i*image_bt.shape[0]:(i+1)*image_bt.shape[0]] = pred_grasp_bt
        true_grasps[i*image_bt.shape[0]:(i+1)*image_bt.shape[0]] = true_grasp_bt

    return pred_grasps, true_grasps

        # if save_path is not None:
        #     raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
        #     draw_annotations(raw_image, generator.load_annotations(i), class_to_bbox_3D = generator.get_bbox_3d_dict(), camera_matrix = generator.load_camera_matrix(i), label_to_name=generator.label_to_name)
        #     draw_detections(raw_image, image_boxes, image_scores, image_labels, image_rotations, image_translations, class_to_bbox_3D = generator.get_bbox_3d_dict(), camera_matrix = generator.load_camera_matrix(i), label_to_name=generator.label_to_name)

        #     cv2.imwrite(os.path.join(save_path, '{}.png'.format(i)), raw_image)


def _get_annotations(generator):
    """ Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = (bboxes = annotations[num_detections, 5], rotations = annotations[num_detections, num_rotation_parameters], translations = annotations[num_detections, num_translation_parameters])

    # Arguments
        generator : The generator used to retrieve ground truth annotations.
    # Returns
        A list of lists containing the annotations for each image in the generator.
    """
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in progressbar.progressbar(range(generator.size()), prefix='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = (annotations['bboxes'][annotations['labels'] == label, :].copy(), annotations['rotations'][annotations['labels'] == label, :].copy(), annotations['translations'][annotations['labels'] == label, :].copy())

    return all_annotations


def check_6d_pose_2d_reprojection(model_3d_points, rotation_gt, translation_gt, rotation_pred, translation_pred, camera_matrix, pixel_threshold = 5.0):
    """ Check if the predicted 6D pose of a single example is considered to be correct using the 2D reprojection metric

    # Arguments
        model_3d_points: numpy array with shape (num_3D_points, 3) containing the object's 3D model points
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        translation_gt: numpy array with shape (3,) containing the ground truth translation vector
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
        translation_pred: numpy array with shape (3,) containing the predicted translation vector
        camera_matrix: numpy array with shape (3, 3) containing the camera matrix
        pixel_threshold: Threshold in pixels when a prdicted 6D pose in considered to be correct
    # Returns
        Boolean indicating wheter the predicted 6D pose is correct or not
    """
    #transform points into camera coordinate system with gt and prediction transformation parameters respectively
    transformed_points_gt = np.dot(model_3d_points, rotation_gt.T) + translation_gt
    transformed_points_pred = np.dot(model_3d_points, rotation_pred.T) + translation_pred
    
    #project the points on the 2d image plane
    points_2D_gt, _ = np.squeeze(cv2.projectPoints(transformed_points_gt, np.zeros((3,)), np.zeros((3,)), camera_matrix, None))
    points_2D_pred, _ = np.squeeze(cv2.projectPoints(transformed_points_pred, np.zeros((3,)), np.zeros((3,)), camera_matrix, None))
    
    distances = np.linalg.norm(points_2D_gt - points_2D_pred, axis = -1)
    mean_distances = np.mean(distances)
    
    if mean_distances <= pixel_threshold:
        is_correct = True
    else:
        is_correct = False
        
    return is_correct


def check_6d_pose_add(model_3d_points, model_3d_diameter, rotation_gt, translation_gt, rotation_pred, translation_pred, diameter_threshold = 0.1):
    """ Check if the predicted 6D pose of a single example is considered to be correct using the ADD metric

    # Arguments
        model_3d_points: numpy array with shape (num_3D_points, 3) containing the object's 3D model points
        model_3d_diameter: Diameter of the object
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        translation_gt: numpy array with shape (3,) containing the ground truth translation vector
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
        translation_pred: numpy array with shape (3,) containing the predicted translation vector
        diameter_threshold: Threshold relative to the object's diameter when a prdicted 6D pose in considered to be correct
    # Returns
        is_correct: Boolean indicating wheter the predicted 6D pose is correct or not
        mean_distances: The average distance between the object's 3D points transformed with the predicted and ground truth 6D pose respectively
        transformed_points_gt: numpy array with shape (num_3D_points, 3) containing the object's 3D points transformed with the ground truth 6D pose
    """
    transformed_points_gt = np.dot(model_3d_points, rotation_gt.T) + translation_gt
    transformed_points_pred = np.dot(model_3d_points, rotation_pred.T) + translation_pred

    distances = np.linalg.norm(transformed_points_gt - transformed_points_pred, axis = -1)
    mean_distances = np.mean(distances)
    
    if mean_distances <= (model_3d_diameter * diameter_threshold):
        is_correct = True
    else:
        is_correct = False
        
    return is_correct, mean_distances, transformed_points_gt


def check_6d_pose_add_s(model_3d_points, model_3d_diameter, rotation_gt, translation_gt, rotation_pred, translation_pred, diameter_threshold = 0.1, max_points = 1000):    
    """ Check if the predicted 6D pose of a single example is considered to be correct using the ADD-S metric

    # Arguments
        model_3d_points: numpy array with shape (num_3D_points, 3) containing the object's 3D model points
        model_3d_diameter: Diameter of the object
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        translation_gt: numpy array with shape (3,) containing the ground truth translation vector
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
        translation_pred: numpy array with shape (3,) containing the predicted translation vector
        diameter_threshold: Threshold relative to the object's diameter when a prdicted 6D pose in considered to be correct
        max_points: Max number of 3D points to calculate the distances (The computed distance between all points to all points can be very memory consuming)
    # Returns
        is_correct: Boolean indicating wheter the predicted 6D pose is correct or not
        mean_distances: The average distance between the object's 3D points transformed with the predicted and ground truth 6D pose respectively
    """
    transformed_points_gt = np.dot(model_3d_points, rotation_gt.T) + translation_gt
    transformed_points_pred = np.dot(model_3d_points, rotation_pred.T) + translation_pred
    #calc all distances between all point pairs and get the minimum distance for every point
    num_points = transformed_points_gt.shape[0]
    
    #approximate the add-s metric and use max max_points of the 3d model points to reduce computational time
    step = num_points // max_points + 1
    
    min_distances = wrapper_c_min_distances(transformed_points_gt[::step, :], transformed_points_pred[::step, :])
    mean_distances = np.mean(min_distances)
    
    if mean_distances <= (model_3d_diameter * diameter_threshold):
        is_correct = True
    else:
        is_correct = False
        
    return is_correct, mean_distances


def calc_translation_diff(translation_gt, translation_pred):
    """ Computes the distance between the predicted and ground truth translation

    # Arguments
        translation_gt: numpy array with shape (3,) containing the ground truth translation vector
        translation_pred: numpy array with shape (3,) containing the predicted translation vector
    # Returns
        The translation distance
    """
    return np.linalg.norm(translation_gt - translation_pred)


def calc_rotation_diff(rotation_gt, rotation_pred):
    """ Calculates the distance between two rotations in degree
        copied and modified from https://github.com/ethnhe/PVN3D/blob/master/pvn3d/lib/utils/evaluation_utils.py
    # Arguments
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
    # Returns
        the rotation distance in degree
    """  
    rotation_diff = np.dot(rotation_pred, rotation_gt.T)
    trace = np.trace(rotation_diff)
    trace = (trace - 1.) / 2.
    if trace < -1.:
        trace = -1.
    elif trace > 1.:
        trace = 1.
    angular_distance = np.rad2deg(np.arccos(trace))
    
    return abs(angular_distance)


def check_6d_pose_5cm_5degree(rotation_gt, translation_gt, rotation_pred, translation_pred):
    """ Check if the predicted 6D pose of a single example is considered to be correct using the 5cm 5 degree metric
        copied and modified from https://github.com/ethnhe/PVN3D/blob/master/pvn3d/lib/utils/evaluation_utils.py def cm_degree_5_metric(self, pose_pred, pose_targets):
    # Arguments
        rotation_gt: numpy array with shape (3, 3) containing the ground truth rotation matrix
        translation_gt: numpy array with shape (3,) containing the ground truth translation vector
        rotation_pred: numpy array with shape (3, 3) containing the predicted rotation matrix
        translation_pred: numpy array with shape (3,) containing the predicted translation vector
    # Returns
        is_correct: Boolean indicating wheter the predicted 6D pose is correct or not
        translation_distance: the translation distance
        rotation_distance: the rotation distance
    """    
    translation_distance = calc_translation_diff(translation_gt, translation_pred)
    
    rotation_distance = calc_rotation_diff(rotation_gt, rotation_pred)
    
    if translation_distance <= 50 and rotation_distance <= 5:
        is_correct = True
    else:
        is_correct = False
        
    return is_correct, translation_distance, rotation_distance


def test_draw(image, camera_matrix, points_3d):
    """ Projects and draws 3D points onto a 2D image and shows the image for debugging purposes

    # Arguments
        image: The image to draw on
        camera_matrix: numpy array with shape (3, 3) containing the camera matrix
        points_3d: numpy array with shape (num_3D_points, 3) containing the 3D points to project and draw (usually the object's 3D points transformed with the ground truth 6D pose)
    """
    points_2D, jacobian = cv2.projectPoints(points_3d, np.zeros((3,)), np.zeros((3,)), camera_matrix, None)
    points_2D = np.squeeze(points_2D)
    points_2D = np.copy(points_2D).astype(np.int32)
    
    tuple_points = tuple(map(tuple, points_2D))
    for point in tuple_points:
        cv2.circle(image, point, 2, (255, 0, 0), -1)
        
    cv2.imshow('image', image)
    cv2.waitKey(0)


def evaluate(
    generator,
    model,
    iou_threshold = 0.5,
    score_threshold = 0.05,
    max_detections = 100,
    save_path = None,
    diameter_threshold = 0.1,
):
    """ Evaluate a given dataset using a given model.

    # Arguments
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection is positive or negative.
        score_threshold: The score confidence threshold to use for detections.
        max_detections: The maximum number of detections to use per image.
        save_path: The path to save images with visualized detections to.
        diameter_threshold: Threshold relative to the object's diameter when a prdicted 6D pose in considered to be correct
    # Returns
        Several dictionaries mapping class names to the computed metrics.
    """
    # gather all detections and annotations
    pred_grasps, true_grasps = _get_detections(generator, model, save_path=save_path)

    # Grasp Loss
    loss_v = []
    min_loss_index = []
    # For an image
    for j in range(len(true_grasps)):
        min_loss = float('inf')
        min_index = 0
        # For a grasp
        for i in range(len(true_grasps[j])):
            cur_loss = grasp_loss(true_grasps[j][i], pred_grasps[j])
            if cur_loss < min_loss:
                min_loss = cur_loss
                min_index = i
        loss_v.append(min_loss)
        min_loss_index.append(i)
    avg_grasp_loss = sum(loss_v) / len(loss_v)

    # IoU Angle Diff
    correct_grasp_count = 0
    iou_list = []
    angle_diff_list = []
    for j in range(len(true_grasps)):
        index = min_loss_index[j]
        bbox_true = grasp_to_bbox( *true_grasps[j][index] )
        bbox_pred = grasp_to_bbox( *pred_grasps[j] )
        
        #IoU
        try:
            p1 = Polygon([bbox_true[0], bbox_true[1], bbox_true[2], bbox_true[3], bbox_true[0]])
            p2 = Polygon([bbox_pred[0], bbox_pred[1], bbox_pred[2], bbox_pred[3], bbox_pred[0]])
            iou = p1.intersection(p2).area / (p1.area +p2.area -p1.intersection(p2).area)
            iou_list.append(iou)
        except Exception as e: 
            print('IoU ERROR', e)
            print('Bbox pred:', bbox_pred)
            print('pred grasp:', pred_grasps[j])
            print('Bbox true:', bbox_true)
        #Angle Diff
        true_sin = true_grasps[j][index][2]
        true_cos = true_grasps[j][index][3]
        if true_cos != 0:
            true_angle = np.arctan(true_sin/true_cos) * 180/np.pi
        else:
            true_angle = 90
        pred_sin = pred_grasps[j][2]
        pred_cos = pred_grasps[j][3]
        if pred_cos != 0:
            pred_angle = np.arctan(pred_sin/pred_cos) * 180/np.pi
        else:
            pred_angle = 90
        angle_diff = np.abs(pred_angle - true_angle)
        angle_diff = min(angle_diff, 180.0 - angle_diff)
        angle_diff_list.append(angle_diff)
        
        if angle_diff < 30. and iou >= 0.25:
            correct_grasp_count += 1
            # print('image: %d | duration = %.2f | count = %d | iou = %.2f | angle_difference = %.2f' %(step, duration, count, iou, angle_diff))
    grasp_accuracy = correct_grasp_count / len(true_grasps)
    avg_iou = sum(iou_list) / len(true_grasps)
    avg_angle_diff = sum(angle_diff_list) / len(true_grasps)

    return avg_grasp_loss, grasp_accuracy, avg_iou, avg_angle_diff

