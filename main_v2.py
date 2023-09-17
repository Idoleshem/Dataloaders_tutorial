import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load the YOLOv5 model
model_path = 'yolov5s.pt'  # Replace with the path to your YOLOv5 model file
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

# Set the input image path
image_path = 'C:\\Users\\Ido\\Desktop\\Dataloaders_tutorial\\frames\\frame_0000.jpg'
# Load the input image using PIL
img = Image.open(image_path)

# Resize the input image to 640x640
#img = img.resize((640, 640), Image.ANTIALIAS)

# Perform inference on the resized image
results = model(img)



# Get the detected objects and their labels
detections = results.pred[0]  # Assuming `results` is your Detections instance
labels = results.names  # List of class labels (e.g., 'person', 'suitcase', 'chair', 'tv')


detections = detections[detections[:, 4] > 0.5]  # Adjust the confidence threshold as needed


# Create a plot of the image
plt.figure(figsize=(12, 8))
plt.imshow(img)

# Loop through the detected objects and draw bounding boxes
for det in detections:
    label, conf, bbox = det[5], det[4], det[:4]
    class_name = labels[int(label)]

    # Extract bounding box coordinates
    x1, y1, x2, y2 = bbox

    # Create a Rectangle patch for the bounding box
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none', label=class_name + f' ({conf:.2f})'
    )

    # Add the bounding box to the plot
    plt.gca().add_patch(rect)

# Remove axis and legend
plt.axis('off')  # Hide axis
plt.legend().set_visible(False)

# Save the image with bounding boxes
output_image_path = 'output_image.jpg'  # Specify the output image file path
plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)

# Close the plot
plt.close()

# Optionally, you can also display a message to confirm that the image has been saved
print(f"Image with bounding boxes saved as '{output_image_path}'")
