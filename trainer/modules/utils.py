import tensorflow as tf
import numpy as np

def iou(box1, box2, is_predict=True):
    """
    Returns the IoU of two bounding boxes
    """
    if is_predict:
       # box coordinate for predict (x,y,w,h)->(xmin,ymin,xmax,ymax)
       pred_box_xmin = box1[..., 0] - box1[..., 2] / 2
       pred_box_ymin = box1[..., 1] - box1[..., 3] / 2
       pred_box_xmax = box1[..., 0] + box1[..., 2] / 2
       pred_box_ymax = box1[..., 1] + box1[..., 3] / 2

       # box coordinate for groud truth (x,y,w,h)->(xmin,ymin,xmax,ymax)
       gt_box_xmin = box2[..., 0] - box2[..., 2] / 2
       gt_box_ymin = box2[..., 1] - box2[..., 3] / 2
       gt_box_xmax = box2[..., 0] + box2[..., 2] / 2
       gt_box_ymax = box2[..., 1] + box2[..., 3] / 2

       # calculate the intersection area
       xmin = tf.maximum(pred_box_xmin, gt_box_xmin)
       ymin = tf.maximum(pred_box_ymin, gt_box_ymin)
       xmax = tf.minimum(pred_box_xmax, gt_box_xmax)
       ymax = tf.minimum(pred_box_ymax, gt_box_ymax)
    
       # Make sure it's a legal intersection
       inter_area = tf.maximum(xmax - xmin, 0) * tf.maximum(ymax - ymin, 0)

       # Calculate the union area
       box1_area = abs((pred_box_xmax - pred_box_xmin) * (pred_box_ymax - pred_box_ymin))
       box2_area = abs((gt_box_xmax - gt_box_xmin) * (gt_box_ymax - gt_box_ymin))
       union_area = box1_area + box2_area - inter_area

       # calculate the iou
       epsilon = 1e-6
       iou = inter_area / (union_area + epsilon)

       return iou

    else:
        raise NotImplementedError


def plot_anchor_boxes(image, anchor_boxes):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(1)
#     print(f'type(image): {type(image)}')
    ax.imshow(image)
#     image_width, image_height = image.size
    image_width, image_height = image.shape[1], image.shape[0]
    print(f'image_width: {image_width}, image_height: {image_height}')
    for anchor_box in anchor_boxes:
        print(f'anchor_box: {anchor_box}')
        x = ( anchor_box[0] - anchor_box[2] / 2 ) * image_width
        y = ( anchor_box[1] - anchor_box[3] / 2 ) * image_height
        w = anchor_box[2] * image_width
        h = anchor_box[3] * image_height
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()  


def labels2grid( imgsize, labels , gridsize, num_classes=1):
    """
    imgsize: (width, height)
    gridsize: (width, height)
    labels: (num_boxes, 5) (x, y, w, h, class)
    num_classes: number of classes

    return: (gridsize*gridsize, 1, 5 + num_classes)
    """

    grid = np.zeros( (gridsize[1]*gridsize[0], 1, 5 + num_classes) )
    grid = grid.astype(np.float32)

    for label in labels:
#         x, y, w, h, c = label # c is class index, y, x, h, w are normalized
        c, x, y, w, h = label # c is class index, y, x, h, w are normalized

#         # Convert box coordinate to grid coordinate
#         xb = int( x * gridsize[0] )
#         yb = int( y * gridsize[1] )
# 
#         # TODO: check this
# #         # Convert box width and height to grid width and height
# #         w = w * gridsize[0]
# #         h = h * gridsize[1]
# 
#         grid_idx = yb * gridsize[0] + xb
#         c = int(c)
#         grid[grid_idx, 0, 0] = x
#         grid[grid_idx, 0, 1] = y
#         grid[grid_idx, 0, 2] = w
#         grid[grid_idx, 0, 3] = h
#         grid[grid_idx, 0, 4] = 1 # confidence score
#         grid[grid_idx, 0, 5 + c] = 1 # class score
#     return grid


        cell_size = 1.0 / gridsize[0]
        xb = int( x // cell_size )
        yb = int( y // cell_size )

        x_cell = x*gridsize[0] - xb
        y_cell = y*gridsize[1] - yb

        grid_idx = yb * gridsize[0] + xb

        c = int(c)
        grid[grid_idx, 0, 0] = x_cell
        grid[grid_idx, 0, 1] = y_cell
        grid[grid_idx, 0, 2] = w
        grid[grid_idx, 0, 3] = h
        grid[grid_idx, 0, 4] = 1 # confidence score
        grid[grid_idx, 0, 5 + c] = 1 # class score
    return grid

# TODO: this works if num_boxes = 1, because the for is wrong
def convert_labels_grid_to_image_scale(grid_labels, grid_size, num_boxes=1):
    image_scale_labels = np.zeros_like(grid_labels)

    for i in range(grid_labels.shape[1]):
        label = grid_labels[0, i]

        confidence = label[4]
        print(f'confidence: {confidence}')
        if confidence > 0:
            x_cell, y_cell, w, h = label[:4]
            class_probs = label[5:]

            grid_x = i // grid_size
            grid_y = i % grid_size

            x = (grid_x + x_cell) / grid_size # maybe is (grid_x - x_cell) / grid_size
            y = (grid_y + y_cell) / grid_size

            image_scale_labels[0, i, 0] = x
            image_scale_labels[0, i, 1] = y
            image_scale_labels[0, i, 2] = w
            image_scale_labels[0, i, 3] = h
            image_scale_labels[0, i, 4] = confidence

            image_scale_labels[0, i, 5:] = class_probs

    return image_scale_labels

def anchor_to_box(image_width, image_height, anchor_box):
    x1 = ( anchor_box[0] - anchor_box[2] / 2 ) * image_width
    y1 = ( anchor_box[1] - anchor_box[3] / 2 ) * image_height
    x2 = ( anchor_box[0] + anchor_box[2] / 2 ) * image_width
    y2 = ( anchor_box[1] + anchor_box[3] / 2 ) * image_height
    return [x1, y1, x2, y2]

def non_maximum_suppression(predictions, confidence_threshold=0.5, iou_threshold=0.5, image_width=416, image_height=416):
    # Define a helper function to calculate IoU (Intersection over Union)
    def calculate_iou(prediction1, prediction2):
        # Calculate intersection coordinates
        box1 = anchor_to_box(image_width, image_height, prediction1)
        box2 = anchor_to_box(image_width, image_height, prediction2)

        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # Calculate intersection area
        intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        # Calculate areas of each box
        area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # Calculate union area
        union = area_box1 + area_box2 - intersection

        # Calculate IoU
        iou = intersection / union
        print(f'iou: {iou}')
        return iou

    # Filter predictions based on confidence threshold
    filtered_predictions = [prediction for prediction in predictions if prediction[4] >= confidence_threshold]

    # Sort predictions by confidence score (probability)
    filtered_predictions.sort(key=lambda x: x[4], reverse=True)

    # Apply Non-Maximum Suppression
    selected_predictions = []
    while len(filtered_predictions) > 0:
        selected_predictions.append(filtered_predictions[0])

        # Remove the selected prediction
        del filtered_predictions[0]

        # Apply NMS
        iou = [calculate_iou(selected_predictions[-1], prediction) for prediction in filtered_predictions]
        print(f'len(iou): {len(iou)}')
        print(f'iou: {iou}')

        filtered_predictions = [prediction for i, prediction in enumerate(filtered_predictions) if iou[i] < iou_threshold]
       
    return selected_predictions



# Test iou function, given a tensor predict box and a ground truth box 
# prediction shape: (batch_size, grid_size*grid_size, num_boxes, 4)
# ground_truth shape: (batch_size, grid_size*grid_size, 1, 4)

if __name__ == '__main__':
    print(f'Tensorflow version: {tf.__version__}')
    print(f'Testing iou function...')
    prediction = tf.random.uniform((1, 7*7, 2, 4))
    ground_truth = tf.random.uniform((1, 7*7, 1, 4))
    iou = iou(prediction, ground_truth)
    assert iou.shape == (1, 7*7, 2)
    print(f'Passed!')

    print(f'Testing labels2grid function...')
    imgpath = '../images/train/road130.png'
    labelpath = '../labels/train/road130.txt'

    from PIL import Image

    im = Image.open(imgpath)
    label = np.loadtxt(labelpath)
    label = label.reshape(-1, 5)

    print(f'label: {label}')

    grid = labels2grid( im.size, label, (4, 4), num_classes=4)

    print(f'grid: {grid}')
