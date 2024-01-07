import utils 
import model
import loss
import dataset

import tensorflow as tf

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
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss_value.numpy()} Accuracy: {epoch_accuracy.result().numpy()}")

    return epoch_loss.result(), epoch_accuracy.result()

if __name__ == '__main__':
    train_dataset = dataset.CustomDataset('../images/train', '../labels/train', grid=(4,4), num_classes=4)
    val_dataset = dataset.CustomDataset('../images/val', '../labels/val', grid=(4,4), num_classes=4)

    anchors = [ [(11,8), (25,18), (46,43)]]

    model = model.YOLO(first_out=16, nc=4, anchors=anchors)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=0.5)
    
    train_loss, train_accuracy = train_one_epoch(model, train_dataset, loss.YOLO_Loss, optimizer, epoch=0)
    print(f"Output Loss: {train_loss}, Accuracy: {train_accuracy}")




