from modules import datasets
from modules import train
from modules import models
from modules import loss


import argparse
import numpy as np
import matplotlib.pyplot as plt
# import mlflow
import logging

import tensorflow as tf

import os 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns

#     logging.getLogger('models.cnn').setLevel(logging.INFO)
#     logging.getLogger('modules.dataset').setLevel(logging.INFO)
#     logging.getLogger('modules.train').setLevel(logging.INFO)
#     logging.getLogger('modules.utils').setLevel(logging.INFO)
#     logging.getLogger('matplotlib').setLevel(logging.WARNING)
# 

# given         dict_log["train_acc"].append(train_accuracy)
#         dict_log["train_loss"].append(train_loss)
#         dict_log["val_acc"].append(val_accuracy)
#         dict_log["val_loss"].append(val_loss)

def plot_logs(dict_log):
    plt.figure(figsize=(20,10))
    plt.subplot(1,2,1)
    plt.plot(dict_log["train_acc"], label="train_acc")
    plt.plot(dict_log["val_acc"], label="val_acc")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(dict_log["train_loss"], label="train_loss")
    plt.plot(dict_log["val_loss"], label="val_loss")
    plt.legend()

#     plt.show()

    # save image
    plt.savefig('results/logs.png')



# plot confusion matrix for YOLO model
def plot_confusion_matrix(model, val_dataset, num_classes, img_size, classes):
    y_true = []
    y_pred = []
    for x, y in val_dataset:
#         print(f'x type {type(x)}, y type {type(y)}')
#         print(f'x shape {x.shape}, y shape {y.shape}')
        y_true.append(y)
        y_hat = model.predict(x)
#         print(f'y_hat shape {y_hat.shape}')
        y_pred.append(y_hat)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    print(f'y_true: {y_true.shape}, y_pred: {y_pred.shape}')

    # Reshape predictions and true labels to flatten the grid dimensions
    predictions_flat = np.reshape(y_pred, (-1, 5 + num_classes))
    true_labels_flat = np.reshape(y_true, (-1, 5 + num_classes))

    # Extract class indices from predictions and true labels
    predicted_class_indices = np.argmax(predictions_flat[:, 5:], axis=1)
    true_class_indices = np.argmax(true_labels_flat[:, 5:], axis=1)
    
    conf_matrix = confusion_matrix(true_class_indices, predicted_class_indices, normalize='true')
    # Display the confusion matrix and save it
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=[classes[i] for i in range(num_classes)])
    disp.plot(cmap=plt.cm.Blues)  # You can choose a colormap that suits your visualization
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
#     plt.show()
    plt.savefig('results/confusion_matrix.png')

    report = classification_report(true_class_indices, predicted_class_indices, target_names=[classes[i] for i in range(num_classes)])

    # save report 
    with open('results/report.txt', 'w') as f:
        f.write(report)

    print(report)


def main( dataset_path, num_classes, grid, num_epochs, batch_size, learning_rate, debug, img_size):
    train_dataset = datasets.CustomDataset(os.path.join(dataset_path, 'images/train'), 
                                  os.path.join(dataset_path, 'labels/train'), 
                                  grid=grid, 
                                  num_classes=num_classes,
                                  batch_size=batch_size,
                                  img_size=img_size)

    val_dataset = datasets.CustomDataset(os.path.join(dataset_path, 'images/val'),
                                os.path.join(dataset_path, 'labels/val'),
                                grid=grid,
                                num_classes=num_classes,
                                batch_size=batch_size,
                                img_size=img_size)

    anchors = [ [(11,8), (25,18), (46,43)]]

    model = models.YOLO(first_out=16, nc=num_classes, anchors=anchors)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=0.5)

    logs = train.train(model, train_dataset, val_dataset, loss.YOLO_Loss, optimizer, num_epochs)

    # plot logs
    plot_logs(logs)
    # save model
    model.save('results/model.keras')
   
    tflite_model = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    with open('results/model.tflite', 'wb') as f:
        f.write(tflite_model)

    classes = {0: 'trafficlight', 1: 'stop', 2: 'speedlimit', 3: 'crosswalk'}

    # load model from file
#     model = tf.keras.models.load_model('model.keras', custom_objects={'YOLO_Loss': loss.YOLO_Loss})
    # load classes
    classes = {0: 'trafficlight', 1: 'stop', 2: 'speedlimit', 3: 'crosswalk'}
    # plot confusion matrix
    plot_confusion_matrix(model, val_dataset, num_classes, img_size, classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='dataset'
                        , help='Dataset path')
    parser.add_argument('--num_classes', type=int, default=4
                        , help='Number of classes')
    parser.add_argument('--grid', type=tuple, default=(4,4)
                        , help='Grid size')
    parser.add_argument('--num_epochs', type=int, default=10
                        , help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16
                        , help='Batch size')
    parser.add_argument('--lr', type=float, default=0.00001
                        , help='Learning rate')
    parser.add_argument('--debug', type=bool, default=False
                        , help='Debug mode')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[160, 160]
                        , help='Image size')
    args = parser.parse_args()


    main(args.dataset, args.num_classes, 
         args.grid, 
         args.num_epochs, 
         args.batch_size, 
         args.lr,
         args.debug, 
         args.imgsz)
