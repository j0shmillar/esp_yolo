import tensorflow as tf
import modules.utils as utils

# y has shape y = tf.reshape(y, (-1, self.grid_size*self.grid_size, self.num_boxes, 5 + nc))

# def YOLO_Loss(y_true, y_pred, anchors, num_classes, lambda_coord=0.5, lambda_noobj=0.5, grid_size=4):
#     num_boxes = len(anchors)
def YOLO_Loss(y_true, y_pred, num_boxes, num_classes, lambda_coord=0.5, lambda_noobj=0.5, grid_size=4):
    y_true = tf.reshape(y_true, (-1, grid_size*grid_size, 1, 5 + num_classes))
    y_pred = tf.reshape(y_pred, (-1, grid_size*grid_size, num_boxes, 5 + num_classes))

    assert y_true.shape[-1] == 5 + num_classes
    assert y_pred.shape[-1] == 5 + num_classes

    # Extract true xywh and confidence
    y_true_xywh = y_true[:, :, :, :4]
    y_true_conf = y_true[:, :, :, 4]
    y_true_class = y_true[:, :, :, 5:]

    assert y_true_xywh.shape[1:] == (grid_size*grid_size, 1, 4)
    assert y_true_conf.shape[1:] == (grid_size*grid_size, 1)

    # Extract predicted xywh and confidence
    y_pred_xywh = y_pred[:, :, :, :4]
    y_pred_conf = y_pred[:, :, :, 4]
    y_pred_class = y_pred[:, :, :, 5:]
    y_pred_xy = y_pred_xywh[:, :, :, :2]
    y_pred_wh = y_pred_xywh[:, :, :, 2:]

    # Shape XYWH = (batch_size, grid_size*grid_size, num_boxes, 4)
    # Shape CONF = (batch_size, grid_size*grid_size, num_boxes, 1)
    assert y_pred_xywh.shape[1:] == (grid_size*grid_size, num_boxes, 4)
    assert y_pred_conf.shape[1:] == (grid_size*grid_size, num_boxes)
    assert y_pred_xy.shape[1:] == (grid_size*grid_size, num_boxes, 2)
    assert y_pred_wh.shape[1:] == (grid_size*grid_size, num_boxes, 2)

    # Extract true objectness and no-objectness masks
    object_mask = y_true_conf
    no_object_mask = 1.0 - object_mask

    # Reshape predictions to calculate objectness and no-objectness
#     y_pred_conf = tf.reshape(y_pred_conf, [-1, grid_size*grid_size, num_boxes, 1])
#     pred_objectness = tf.sigmoid(y_pred_conf)  # TODO: check this
    pred_objectness = y_pred_conf
#     pred_objectness = y_pred_conf

    # Split true and predicted xy, wh
    y_true_xy = y_true_xywh[:, :, :, :2] # shape: (batch_size, grid_size*grid_size, 1, 2)
    y_true_wh = y_true_xywh[:, :, :, 2:] # shape: (batch_size, grid_size*grid_size, 1, 2)
    y_pred_xy = y_pred_xywh[:, :, :, :2] # shape: (batch_size, grid_size*grid_size, num_boxes, 2)
    y_pred_wh = y_pred_xywh[:, :, :, 2:] # shape: (batch_size, grid_size*grid_size, num_boxes, 2)

    # Calculate box differences (predicted - true)
    box_diff = y_pred_xy - y_true_xy
    box_diff = tf.concat([box_diff, y_pred_wh - y_true_wh], axis=-1) # shape: (batch_size, grid_size*grid_size, num_boxes, 4)
#     sqrt_true_wh = tf.sqrt(y_true_wh)
#     sqrt_pred_wh = tf.sqrt(y_pred_wh) # Why here is NaN?
#     box_diff = tf.concat([box_diff, sqrt_pred_wh - sqrt_true_wh], axis=-1) # shape: (batch_size, grid_size*grid_size, num_boxes, 4)

    assert box_diff.shape[1:] == (grid_size*grid_size, num_boxes, 4)

    # Calculate coordinate loss
    xy_loss = lambda_coord * object_mask * tf.reduce_sum(tf.square(box_diff[..., :2]), axis=-1)
    wh_loss = lambda_coord * object_mask * tf.reduce_sum(tf.square(box_diff[..., 2:]), axis=-1)
    assert xy_loss.shape[1:] == (grid_size*grid_size, num_boxes)

    # Calculate IOU between predicted and true boxes, my IOU function accepts xywh of two boxes,
    # so how to calculate IOU between true and predicted boxes?
    iou = utils.iou(y_pred_xywh, y_true_xywh) # shape: (batch_size, grid_size*grid_size, num_boxes)
    assert iou.shape[1:] == (grid_size*grid_size, num_boxes)

    # Calculate objectness loss and no-objectness loss
    obj_loss = object_mask * tf.square(pred_objectness - iou)
    noobj_loss = lambda_noobj * no_object_mask * tf.square(pred_objectness - iou)
    assert obj_loss.shape[1:] == (grid_size*grid_size, num_boxes)
    assert noobj_loss.shape[1:] == (grid_size*grid_size, num_boxes)

    # Calculate classification loss
    class_loss = object_mask * tf.reduce_sum(tf.square(y_true_class - y_pred_class), axis=-1)
    assert class_loss.shape[1:] == (grid_size*grid_size, num_boxes)

    # Compute total loss
    total_loss = xy_loss + wh_loss + obj_loss + noobj_loss + class_loss
    total_loss = tf.reduce_sum(total_loss)

    return total_loss

# Test loss function with dummy data, assuming batch_size = 1, grid_size = 4, num_boxes = 2, num_classes = 3

if __name__ == '__main__':
    import numpy as np
    print(f'TensorFlow version: {tf.__version__}')
    num_classes = 4
    print('Test loss function with dummy data, assuming batch_size = 1, grid_size = 4, num_boxes = 2, num_classes = {}'.format(num_classes))
    y_true = np.random.rand(1, 4*4, 5 + num_classes)
    y_pred = np.random.rand(1, 4*4, 2, 5 + num_classes)
    anchors = [(1, 1), (2, 2)]
    num_boxes = len(anchors)
    loss = YOLO_Loss(y_true, y_pred, num_boxes, num_classes)
    print(loss)

