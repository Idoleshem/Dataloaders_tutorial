import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def save_image_with_bounding_boxes(image_path, bounding_boxes, output_image_path):
    # Load the YOLOv5 model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    # Load the input image using PIL
    img = Image.open(image_path)

    # Resize the input image to 640x640
    img = img.resize((640, 640), Image.ANTIALIAS)

    # Create a plot of the image
    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    # Loop through the detected objects and draw bounding boxes
    for bbox in bounding_boxes:
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none'
        )
        plt.gca().add_patch(rect)

    # Remove axis
    plt.axis('off')

    # Save the image with bounding boxes
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)

    # Close the plot
    plt.close()

