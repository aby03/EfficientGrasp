import argparse
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam#, Adam_accumulate

from model import build_EfficientGrasp
# from model_split import build_EfficientGrasp
from losses import smooth_l1, focal, transformation_loss, grasp_loss_bt
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES

from custom_load_weights import custom_load_weights
import json

# from generators.cornell import CornellGenerator
from dataset_processing.cornell_data import CornellDataset

def parse_args(args):
    """
    Parse the arguments.
    """
    date_and_time = time.strftime("%d_%m_%Y_%H_%M_%S")
    parser = argparse.ArgumentParser(description = 'Simple EfficientGrasp training script.')
    subparsers = parser.add_subparsers(help = 'Arguments for specific dataset types.', dest = 'dataset_type')
    subparsers.required = True
    
    cornell_parser = subparsers.add_parser('cornell')
    cornell_parser.add_argument('cornell_path', help = 'Path to dataset directory (ie. /Datasets/Linemod_preprocessed/).')

    parser.add_argument('--weights', help = 'File containing weights to init the model parameter')
    parser.add_argument('--freeze-backbone', help = 'Freeze training of backbone layers.', action = 'store_true')
    parser.add_argument('--no-freeze-bn', help = 'Do not freeze training of BatchNormalization layers.', action = 'store_true')

    parser.add_argument('--batch-size', help = 'Size of the batches.', default = 1, type = int)
    parser.add_argument('--lr', help = 'Learning rate', default = 1e-4, type = float)
    parser.add_argument('--phi', help = 'Hyper parameter phi', default = 0, type = int, choices = (0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--gpu', help = 'Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--epochs', help = 'Number of epochs to train.', type = int, default = 100)
    parser.add_argument('--start-epoch', help = 'Epoch count to start for resuming training', dest = 'start_epoch', type = int, default = 0)
    parser.add_argument('--steps', help = 'Number of steps per epoch.', type = int, default = int(179 * 10))
    parser.add_argument('--snapshot-path', help = 'Path to store snapshots of models during training', default = os.path.join("checkpoints", date_and_time))
    parser.add_argument('--tensorboard-dir', help = 'Log directory for Tensorboard output', default = os.path.join("logs", date_and_time))
    parser.add_argument('--no-snapshots', help = 'Disable saving snapshots.', dest = 'snapshots', action = 'store_false')
    parser.add_argument('--no-evaluation', help = 'Disable per epoch evaluation.', dest = 'evaluation', action = 'store_false')
    parser.add_argument('--compute-val-loss', help = 'Compute validation loss during training', dest = 'compute_val_loss', action = 'store_true')
    parser.add_argument('--score-threshold', help = 'score threshold for non max suppresion', type = float, default = 0.5)
    parser.add_argument('--validation-image-save-path', help = 'path where to save the predicted validation images after each epoch', default = None)

    # Fit generator arguments
    parser.add_argument('--multiprocessing', help = 'Use multiprocessing in fit_generator.', action = 'store_true')
    parser.add_argument('--workers', help = 'Number of generator workers.', type = int, default = 4)
    parser.add_argument('--max-queue-size', help = 'Queue length for multiprocessing workers in fit_generator.', type = int, default = 10)
    
    print(vars(parser.parse_args(args)))
    return parser.parse_args(args)


def main(args = None):
    """
    Train an EfficientGrasp model.

    Args:
        args: parseargs object containing configuration for the training procedure.
    """
    
    allow_gpu_growth_memory()
    
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create the generators
    print("\nCreating the Generators...")
    train_generator, validation_generator = create_generators(args)
    print("Done!")
    
    num_anchors = 1

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    print("\nBuilding the Model...")
    model, prediction_model, all_layers = build_EfficientGrasp(args.phi,
                                                              num_anchors = num_anchors,
                                                              freeze_bn = not args.no_freeze_bn,
                                                              print_architecture=False)

    print("Done!")

    # load pretrained weights
    if args.weights:
        if args.weights == 'imagenet':
            model_name = 'efficientnet-b{}'.format(args.phi)
            file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(model_name)
            file_hash = WEIGHTS_HASHES[model_name][1]
            weights_path = keras.utils.get_file(file_name,
                                                BASE_WEIGHTS_PATH + file_name,
                                                cache_subdir='models',
                                                file_hash=file_hash)
            model.load_weights(weights_path, by_name=True)
        else:
            print('Loading model, this may take a second...')
            custom_load_weights(filepath = args.weights, layers = all_layers, skip_mismatch = True)
            print("\nCustom Weights Loaded!")

    # freeze backbone layers
    if args.freeze_backbone:
        # 227, 329, 329, 374, 464, 566, 656
        for i in range(1, [227, 329, 329, 374, 464, 566, 656][args.phi]):
            model.layers[i].trainable = False

    # mse = tf.keras.losses.MeanSquaredError()
    
    # compile model    
    # Default Adam optimizer
    model.compile(optimizer=Adam(lr = args.lr, clipnorm = 0.001),
                  loss={'regression': grasp_loss_bt(args.batch_size)})
                #   loss={'regression': mse})

    # # Accumulate adam optimizer
    # custom_adam = Adam_accumulate(lr=args.lr, accum_iters=16)
    # model.compile(optimizer=custom_adam, 
    #               loss={'regression': mse})

    # create the callbacks
    callbacks = create_callbacks(
        model,
        prediction_model,
        validation_generator,
        args,
    )

    if not args.compute_val_loss:
        validation_generator = None
    elif args.compute_val_loss and validation_generator is None:
        raise ValueError('When you have no validation data, you should not specify --compute-val-loss.')

    # ## TEST ON SINGLE IMAGE
    # import numpy as np
    # filename = '/home/aby/Workspace/MTP/Datasets/Cornell/archive/06/pcd0600r.png'
    # from generators.cornell import load_and_preprocess_img
    # test_data = load_and_preprocess_img(filename, side_after_crop=None, resize_height=512, resize_width=512)
    # test_data = np.array(test_data)
    # test_data = test_data[np.newaxis, ...]
    # print(' ### TEST ###: ', test_data.shape)
    # test_out = model.predict(test_data, verbose=1, steps=1)
    # print(len(test_out))
    # print(type(test_out[0]))
    # # print(model.layers['grasp_5'].output)
    # print(test_out.shape)
    # # print(test_out[1].shape)
    # exit()


    TEST = True
    if TEST:
        # start testing
        model.fit_generator(
            generator = train_generator,
            steps_per_epoch = 1,
            initial_epoch = 0,
            epochs = args.epochs,
            verbose = 1,
            callbacks = callbacks,
            workers = args.workers,
            use_multiprocessing = args.multiprocessing,
            max_queue_size = args.max_queue_size,
            validation_data = validation_generator
        )
    else:
        # start training
        model.fit_generator(
            generator = train_generator,
            steps_per_epoch = len(train_generator),
            initial_epoch = args.start_epoch,
            epochs = args.epochs,
            verbose = 1,
            callbacks = callbacks,
            workers = args.workers,
            use_multiprocessing = args.multiprocessing,
            max_queue_size = args.max_queue_size,
            validation_data = validation_generator
        )
    os.makedirs(args.snapshot_path, exist_ok = True)
    model.save(os.path.join(args.snapshot_path, '{dataset_type}_finish.h5'.format(dataset_type = args.dataset_type)))
    return

def allow_gpu_growth_memory():
    """
        Set allow growth GPU memory to true

    """
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    _ = tf.compat.v1.Session(config = config)


def create_callbacks(training_model, prediction_model, validation_generator, args):
    """
    Creates the callbacks to use during training.

    Args:
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None
    
    if args.dataset_type == "cornell":
        snapshot_path = args.snapshot_path
        save_path = args.validation_image_save_path
        tensorboard_dir = args.tensorboard_dir
        metric_to_monitor = "grasp_accuracy"
        mode = "max"
    else:
        snapshot_path = args.snapshot_path
        save_path = args.validation_image_save_path
        tensorboard_dir = args.tensorboard_dir
        
    if save_path:
        os.makedirs(save_path, exist_ok = True)

    if tensorboard_dir:
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir = tensorboard_dir,
            histogram_freq = 1,
            write_graph = True,
            write_grads = False,
            write_images = False,
            embeddings_freq = 0,
            embeddings_layer_names = None,
            embeddings_metadata = None,
            profile_batch = 2
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        from eval.eval_callback import Evaluate
        evaluation = Evaluate(validation_generator, prediction_model, tensorboard = tensorboard_callback, save_path = save_path)
        callbacks.append(evaluation)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        os.makedirs(snapshot_path, exist_ok = True)
        # checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(snapshot_path, 'phi_{phi}_{dataset_type}_best_{metric}.h5'.format(phi = str(args.phi), metric = metric_to_monitor, dataset_type = args.dataset_type)),
        checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(snapshot_path, '{dataset_type}_best_{metric}.h5'.format(phi = str(args.phi), metric = metric_to_monitor, dataset_type = args.dataset_type)),
                                                     verbose = 1,
                                                     #save_weights_only = True,
                                                     save_best_only = True,
                                                     monitor = metric_to_monitor,
                                                     mode = mode)
        callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor    = 'loss',
        factor     = 0.5,
        patience   = 10,
        verbose    = 1,
        mode       = 'min',
        min_delta  = 0.0001,
        cooldown   = 0,
        min_lr     = 1e-7
    ))

    return callbacks


def create_generators(args):
    """
    Create generators for training and validation.

    Args:
        args: parseargs object containing configuration for generators.
    Returns:
        The training and validation generators.
    """
    common_args = {
        'batch_size': args.batch_size,
        'phi': args.phi,
    }
    
    if args.dataset_type == 'cornell':
        dataset = args.cornell_path

        # open output file for reading
        with open(dataset+'/train_1.txt', 'r') as filehandle:
            train_data = json.load(filehandle)
        with open(dataset+'/valid_1.txt', 'r') as filehandle:
            valid_data = json.load(filehandle)
        
        # # Shuffle the list of image paths
        # np.random.shuffle(train_data)

        train_generator = CornellDataset(
            dataset,
            train_data,
            **common_args
        )

        validation_generator = CornellDataset(
            dataset,
            valid_data,
            train=False,
            **common_args
        )
    else:
        raise ValueError('Invalid data type received: {}'.format(args.dataset_type))

    return train_generator, validation_generator


if __name__ == '__main__':
    main()
