import modules.utils as utils
import modules.models as models
import modules.loss as loss
import modules.datasets as datasets

import tensorflow as tf
from tqdm import tqdm

def train_one_epoch( model, train_dataset, loss_fn, optimizer, epoch ):
    epoch_loss = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            prediction = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, prediction, num_boxes = model.num_boxes, num_classes = model.nc)

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        epoch_loss(loss_value)
        epoch_accuracy(y_batch_train, prediction)

        if step % 10 == 0:
            # print losses and accuraty with 6 decimals
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss_value.numpy():.6f} Accuracy: {epoch_accuracy.result().numpy():.6f}")
#             print(f"Epoch: {epoch}, Step: {step}, Loss: {loss_value.numpy()} Accuracy: {epoch_accuracy.result().numpy()}")

    return epoch_loss.result(), epoch_accuracy.result()


def train( model, train_dataset, val_dataset, loss_fn, optimizer, epochs ):

    dict_log = {"train_acc":[], "train_loss":[], "val_acc":[], "val_loss":[]}

    progress_bar = tqdm(range(epochs)) 
    for epoch in progress_bar:
        train_loss, train_accuracy = train_one_epoch(model, train_dataset, loss_fn, optimizer, epoch)
        val_loss, val_accuracy = train_one_epoch(model, val_dataset, loss_fn, optimizer, epoch)

        dict_log["train_acc"].append(train_accuracy)
        dict_log["train_loss"].append(train_loss)
        dict_log["val_acc"].append(val_accuracy)
        dict_log["val_loss"].append(val_loss)

        progress_bar.set_description(f"Epoch: {epoch}, Train Loss: {train_loss.numpy():.6f}, Train Accuracy: {train_accuracy.numpy():.6f}, Val Loss: {val_loss.numpy():.6f}, Val Accuracy: {val_accuracy.numpy():.6f}")

#         progress_bar.set_description(f"Epoch: {epoch}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")

    return dict_log

if __name__ == '__main__':
    train_dataset = datasets.CustomDataset('../images/train', '../labels/train', grid=(4,4), num_classes=4)
    val_dataset = datasets.CustomDataset('../images/val', '../labels/val', grid=(4,4), num_classes=4)

    anchors = [ [(11,8), (25,18), (46,43)]]

    model = models.YOLO(first_out=16, nc=4, anchors=anchors)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=0.5)
    
    train_loss, train_accuracy = train_one_epoch(model, train_dataset, loss.YOLO_Loss, optimizer, epoch=0)
    print(f"Output Loss: {train_loss}, Accuracy: {train_accuracy}")




