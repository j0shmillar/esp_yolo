# create a custom dataset.. the images are under the folder 'images', and the labels are under the folder 'labels'. Both folders have their own subfolders. train, val, test.  images are in .png format, and labels are in .txt format.  for example images/train/1.png and labels/train/1.txt

import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
import utils

class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, label_dir, batch_size=32, img_size=(160,160), shuffle=True, grid=(5,5), num_classes=1):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.grid = grid
        self.num_classes = num_classes

        self.image_path = os.path.join(self.image_dir, '*.png')
        self.label_path = os.path.join(self.label_dir, '*.txt')
        self.image_filenames = glob.glob(self.image_path)
        self.label_filenames = glob.glob(self.label_path)
        print(f"Found {len(self.image_filenames)} images and {len(self.label_filenames)} labels")

        self.image_filenames.sort()
        self.label_filenames.sort()

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.label_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = []
        labels = []

        for i in range(len(batch_x)):
            image = Image.open(batch_x[i])
            if ( image.mode != 'RGB' ):
                image = image.convert('RGB')
            image = image.resize(self.img_size) 
            image = np.array(image)
            image = image / 255.0

            label = np.loadtxt(batch_y[i])
            label = label.reshape(-1, 5)
            
            labeled_grid = utils.labels2grid( image.shape[0:2], label, self.grid, self.num_classes)

            images.append(image)
            labels.append(labeled_grid)

        images = np.array(images)
        labels = np.array(labels) 

        return images, labels

    def on_epoch_end(self):
        if self.shuffle:
            self.image_filenames, self.label_filenames = shuffle(self.image_filenames, self.label_filenames)


if __name__ == '__main__':
    grid = (5,5)
    num_classes = 4
    train_dataset = CustomDataset('../images/train', '../labels/train', grid=grid, num_classes=num_classes)
    val_dataset = CustomDataset('../images/val', '../labels/val', grid=grid, num_classes=num_classes)

    # Get images and labels from the first batch
    train_images, train_labels = train_dataset[0]
    val_images, val_labels = val_dataset[0]

    print(f"train_images.shape: {train_images.shape}")
    print(f"train_labels.shape: {train_labels.shape}")
    assert train_images.shape == (32, 160, 160, 3)
    assert train_labels.shape == (32, grid[0]*grid[1], 1, 5 + num_classes)


