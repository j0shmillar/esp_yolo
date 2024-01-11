from modules import datasets
from modules import models
from modules import loss
from modules import utils

import tensorflow as tf
import numpy as np
import os
import argparse

# CONF_THRESHOLD = 0.5
CONF_THRESHOLD = 0.05
IOU_THRESHOLD = 0.5

def main(dataset, num_classes, imgsz, model):
    # load test dataset
    test_dataset = datasets.CustomDataset(os.path.join(dataset, 'images/val'),
                                    os.path.join(dataset, 'labels/val'),
                                    num_classes=num_classes,
                                    img_size=imgsz)
    
    # load model from file
    model = tf.keras.models.load_model(model, custom_objects={'YOLO_Loss': loss.YOLO_Loss})
    # load classes
    classes = {0: 'trafficlight', 1: 'stop', 2: 'speedlimit', 3: 'crosswalk'}

    y_true = []
    y_pred = []
    for x, y in test_dataset:
        y_pred = model.predict(x)
        # y_pred shape [batch_size, grid*grid, num_anchors, 5 + num_classes]
        # y_true shape [batch_size, grid*grid, 1    , 5 + num_classes]
        print(f'y_pred shape: {y_pred.shape}')

        # reshape y_pred to [batch_size, 1, grid*grid*num_anchors, 5 + num_classes]
        y_pred = np.reshape(y_pred, (y_pred.shape[0], 1, y_pred.shape[1]*y_pred.shape[2], y_pred.shape[3]))
        print(f'y_pred shape before NMS: {y_pred.shape}')
        
        # plot anchor boxes for each image in batch
        for i in range(x.shape[0]):
            pred = y_pred[i,:,:,:]
            print(f'pred.shape: {pred.shape}')
            pred = utils.convert_labels_grid_to_image_scale(pred, grid_size = 4)
            print(f'pred.shape after : {pred.shape}')
#             candidates = pred[pred[:,:, 4] >= CONF_THRESHOLD]
            candidates = pred
            print(f'candidates.shape: {candidates.shape}')
            if candidates.shape[0] == 0:
                print('No candidates found')
                continue

            for j in range(candidates.shape[0]):
                print(f'candidates[{j}]: {candidates[j]}')
                utils.plot_anchor_boxes(x[i], candidates[j])
            # pred is a list
#             pred = utils.non_maximum_suppression(candidates, CONF_THRESHOLD, IOU_THRESHOLD, imgsz[0], imgsz[1])
            print(f'pred.shape: {np.array(pred).shape}')
#             utils.plot_anchor_boxes(x[i], pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset'
                        , help='Dataset path')
    parser.add_argument('--num_classes', type=int, default=4
                        , help='Number of classes')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[88,88]
                        , help='Image size')
    parser.add_argument('--model', type=str, default='results/model.keras'
                        , help='Model path')
    args = parser.parse_args()

    main(args.dataset, args.num_classes, args.imgsz, args.model)
