
import torch
import os
import cv2
import pandas as pd
from helpers import extract_frames
def process_video(video_path):

    print("start extracting frames")

    frames_dir = "C:\\Users\\Ido\\Desktop\\Dataloaders_tutorial\\frames"
    #frames_dir = extract_frames(video_path)

    frame_files = os.listdir(frames_dir)    
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    
    #prediction = model 

    # Inference
    results = model(first_frame)

    results.pandas().xyxy[0]

if __name__ == "__main__":
    video_path = "data/bus.mp4"

    total_persons = process_video(video_path)

    print(f"Total persons detected: {total_persons}")