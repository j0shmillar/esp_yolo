import numpy as np
import tensorflow as tf
from PIL import Image
import sys

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

def estimate_memory_usage(tflite_float_model):
    interpreter = tf.lite.Interpreter(model_content=tflite_float_model)
    interpreter.allocate_tensors()

    flash_size = len(tflite_float_model)
    print(f'flash_size (kilobytes): {flash_size / 1024}')
    
    input_tensor = interpreter.tensor(interpreter.get_input_details()[0]['index'])
    input_size = np.prod(input_tensor().shape) * input_tensor().dtype.itemsize

    output_tensor = interpreter.tensor(interpreter.get_output_details()[0]['index'])
    output_size = np.prod(output_tensor().shape) * output_tensor().dtype.itemsize

    activation_size = 0
    # Iterate through layers to estimate activation memory usage
    for i in range(interpreter.get_tensor_details().__len__()):
        tensor_details = interpreter.get_tensor_details()[i]
        tensor_shape = tensor_details['shape']
        tensor_type = tensor_details['dtype']

        tensor_size = np.prod(tensor_shape) * np.dtype(tensor_type).itemsize

#         activation_size += tensor_size

#         print(f'tensor_details name: {tensor_details["name"]}')
        if tensor_details['name'].startswith('model'):  # Assuming 'sequential' prefix for layers
                activation_size += tensor_size

    ram_size = input_size + output_size  + activation_size
    print(f'RAM Summary:')
    print(f'input_size (kilobytes): {input_size / 1024}')
    print(f'output_size (kilobytes): {output_size / 1024}')
    print(f'activation_size (kilobytes): {activation_size / 1024}')
    print(f'ram_size (kilobytes): {ram_size / 1024}')

def convert_quantized_model_to_tflite(quantized_model_path, tflite_model_path):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(quantized_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()

    # Save the model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_quant_model)

    print('TFLite model saved!')

def convert_float_model_to_tflite(float_model_path, tflite_model_path):
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(float_model_path)
    tflite_float_model = converter.convert()

    # Save the model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_float_model)

    print('TFLite model saved!')

def plot_anchor_boxes(image, anchor_boxes):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    image_width, image_height = image.size
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


# input_image_path = 'bw/test/road316.png'
input_image_path = 'images/test/road316.png'
# model_path='../runs/train/yolo_road_det78/weights/best_saved_model'

model_path=sys.argv[1]

# load tf model
model = tf.saved_model.load(model_path)

tflite_model = convert_float_model_to_tflite(model_path, 'float_model.tflite')
quantized_tflite_model = convert_quantized_model_to_tflite(model_path, 'quantized_model.tflite')

image = Image.open(input_image_path)

image_width, image_height = image.size

image = image.resize((160, 160), Image.NEAREST)
print(f'image size of resized image: {image.size} with color channels: {image.mode}')
# convert image to numpy array
image_data = np.asarray(image)
if ( image.mode == 'RGBA' ):
    image_data = image_data[:, :, :3]
print(f'image_data.shape: {image_data.shape}')
image_data = image_data.astype(np.float32)
image_data = image_data / 255.0
print(f'image_data.shape: {image_data.shape}')
image_data = np.expand_dims(image_data, axis=0)

print(f'image_data.shape: {image_data.shape}')

# run inference
output_data = model(image_data)

# convert tuple to numpy array
output_data = np.array(output_data)
print(f'output_data.shape: {output_data.shape}')

# shape 1x1x19200x9
# delete all predictions with low confidence score
# YOLO = [ x, y, w, h, confidence, class1, class2, ... ]
candidates = output_data[output_data[:, :, :, 4] >= CONF_THRESHOLD]
print(f'candidates.shape: {candidates.shape}')
print(f'classes predicted: {candidates[:, 5:]}')

pred = non_maximum_suppression(candidates, CONF_THRESHOLD, IOU_THRESHOLD, image_width, image_height)

print(f'pred.shape: {np.array(pred).shape}')
plot_anchor_boxes( Image.open(input_image_path), pred)



