from PIL import Image
import os

# Function to convert image to grayscale and resize
def process_image(input_path, output_path, size):
    img = Image.open(input_path)
    img = img.resize((size, size), Image.NEAREST)  # Resize image
    img.save(output_path)  # Save processed image

# Function to update YOLO labels
def update_labels(label_path, size):
    with open(label_path, 'r') as file:
        content = file.readlines()
        
    # Update YOLO labels (class x y bw bh)
    for i, line in enumerate(content):
        values = line.strip().split()
        x, y, w, h = map(float, values[1:])
        
        # Calculate new values based on resized image
        x *= size
        y *= size
        w *= size
        h *= size
        
        content[i] = f"{values[0]} {x} {y} {w} {h}\n"

    with open(label_path, 'w') as file:
        file.writelines(content)

# Define input and output directories
image_dir = 'images'
label_dir = 'labels'
sizes = 160 # NxN size for the images

# Process images in train, test, and val directories
for directory in ['train', 'test', 'val']:
    image_folder = os.path.join(image_dir, directory)
    label_folder = os.path.join(label_dir, directory)

    # Process each image in the directory
    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            # Process image
            image_path = os.path.join(image_folder, filename)
            processed_image_path = os.path.join(image_folder, f'processed_{filename}')
            process_image(image_path, processed_image_path, sizes)

#             label_path = os.path.join(label_folder, filename.replace('.png', '.txt'))
#             update_labels(label_path, sizes)

            # Replace original image with processed one
            os.remove(image_path)
            os.rename(processed_image_path, image_path)

print("Images processed and labels updated successfully!")

