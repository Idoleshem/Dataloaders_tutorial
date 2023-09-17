import os

import cv2
import pandas as pd
import torch
from torchvision.ops import nms


def extract_frames(video_path):
    # Create a 'frames' subfolder if it doesn't exist
    output_dir = "frames"
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        
        print(f"extract frame {frame_count}")
        ret, frame = cap.read()

        # Break the loop when the video ends
        if not ret:
            break

        # Save the frame in the 'frames' subfolder
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()

    return os.path.abspath(output_dir)  # Return the absolute path to the 'frames' subdirectory



def nms(detections, confidence_threshold=0.7, iou_threshold=0.5):
    # Sort detections by confidence score in descending order
    detections = sorted(detections, key=lambda x: x[4], reverse=True)

    selected_detections = []

    while detections:
        current_detection = detections[0]
        selected_detections.append(current_detection)

        detections = detections[1:]

        # Calculate IoU with the current detection for all remaining detections
        iou_values = [calculate_iou(current_detection, detection) for detection in detections]

        # Remove detections with high IoU
        detections = [detection for i, detection in enumerate(detections) if iou_values[i] <= iou_threshold]

    return selected_detections

def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate areas of both boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate IoU
    iou = intersection_area / float(area1 + area2 - intersection_area)

    return iou

def extract_batch_results(result_df, results, frame_idx, frame_number):
    print(f"frame num - {frame_number}")
    confidence_threshold = 0.7
    nms_threshold = 0.95  # Adjust as needed

    # Filter detections based on confidence threshold
    detections = results[frame_idx, results[frame_idx, :, 4] > confidence_threshold]

    # Extract class probabilities and class labels
    class_probs = detections[:, 5:]
    class_scores = torch.softmax(class_probs, dim=1)
    predicted_classes = torch.argmax(class_scores, dim=1)
    confidences = detections[:, 4]

    # Apply NMS to filter overlapping bounding boxes
    keep_indices = nms(detections[:, :5], confidence_threshold, 0.5)
    selected_indices = []  # Initialize a list to store the selected box indices

    # Iterate over the filtered detections after NMS
    for box_idx in keep_indices:
        predicted_class = predicted_classes[box_idx].item()
        confidence = confidences[box_idx].item()
        bbox = detections[box_idx, :4].tolist()

        # Check if the predicted class is not class 0
        if predicted_class == 0:
            # Add the results to the DataFrame
            result_df.loc[len(result_df)] = [frame_number, int(box_idx), predicted_class, confidence, bbox[0], bbox[1], bbox[2], bbox[3]]
            
            # Add the box index to the selected_indices list
            selected_indices.append(int(box_idx))

    return result_df,   



# def extract_batch_results(result_df,results,frame_idx,frame_number):
  
#     print(f"frame num - {frame_number}")
#     confidence_threshold = 0.7
#     for box_idx in range(results.size(1)):  # Iterate over the boxes
#         class_probs = results[frame_idx, box_idx, 5:]  # Start at index 5 for class probabilities
#         class_scores = torch.softmax(class_probs, dim=0)
#         predicted_class = torch.argmax(class_scores).item()
#         confidence = results[frame_idx, box_idx, 4].item()

#         # Check if the predicted class is not class 0
#         if predicted_class == 0 and confidence >= confidence_threshold:

#             bbox = results[frame_idx, box_idx, :4].tolist()
            
#             # Add the results to the DataFrame
#             result_df.loc[len(result_df)] = [frame_number, box_idx, predicted_class, confidence, bbox[0], bbox[1], bbox[2], bbox[3]]   



#     return result_df





def visualize_yolo_inference(result_df, frames_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    original_image_size = [640,640]#[1080,1920]
    width_scale = original_image_size[0] / 640
    height_scale = original_image_size[1] / 640

    # Iterate through unique frame numbers
    for frame_number in result_df.index.get_level_values(0).unique():
        frame_number_str = f"{int(frame_number):04d}"

        # Initialize an empty image for the frame
        frame_image = None

        # Iterate through rows for the current frame and draw bounding boxes
        for _, row in result_df.loc[frame_number].iterrows():
            x_min, y_min, x_max, y_max = row['x_min'], row['y_min'], row['x_max'], row['y_max']
            x_min = int(x_min * width_scale)
            y_min = int(y_min * height_scale)
            x_max = int(x_max * width_scale)
            y_max = int(y_max * height_scale)
            # Load the corresponding image or create an empty one if it doesn't exist
            if frame_image is None:
                image_path = os.path.join(frames_dir, f"frame_{frame_number_str}.jpg")
                frame_image = cv2.imread(image_path)
                #frame_image = cv2.resize(frame_image, (640, 640))

            # Draw the bounding box on the image
            color = (0, 255, 0)  # Green color for the bounding box
            thickness = 4  # Thickness of the bounding box
            cv2.rectangle(frame_image, (x_min, y_min), (x_max, y_max), color, thickness)

        # Save the image with all the bounding boxes to the output directory
        output_image_path = os.path.join(output_dir, f"frame_{frame_number_str}_bbox.jpg")
        cv2.imwrite(output_image_path, frame_image)