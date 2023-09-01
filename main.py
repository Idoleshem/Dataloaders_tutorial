
import os
import warnings
import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from helpers import extract_frames

# Suppress DeprecationWarnings related to pandas
warnings.filterwarnings("ignore", category=DeprecationWarning)

class FrameDataset(Dataset):
    def __init__(self, frames_dir, transform=None):
        self.frames_dir = frames_dir
        self.frame_files = os.listdir(frames_dir)
        self.transform = transform

    def __len__(self):
        return len(self.frame_files)

    def __getitem__(self, idx):
        frame_file = self.frame_files[idx]
        frame_path = os.path.join(self.frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        
        if self.transform:
            frame = self.transform(frame)
        
        return frame

def extract_batch_results(result_df,results,batch_idx,frame_number):
    confidence_threshold = 0.7

    print(f"frame num - {frame_number}")

    for box_idx in range(results.size(1)):  # Iterate over the boxes
        class_probs = results[batch_idx, box_idx, 5:]  # Start at index 5 for class probabilities
        class_scores = torch.softmax(class_probs, dim=0)
        predicted_class = torch.argmax(class_scores).item()
        confidence = results[batch_idx, box_idx, 4].item()

        # Check if the predicted class is not class 0
        if predicted_class == 0 and confidence >= confidence_threshold:

            bbox = results[batch_idx, box_idx, :4].tolist()
            
            # Add the results to the DataFrame
            result_df.loc[len(result_df)] = [frame_number, box_idx, predicted_class, confidence, bbox[0], bbox[1], bbox[2], bbox[3]]   

    return result_df

def process_video(video_path,batch_size):

    frames_dir = "C:\\Users\\Ido\\Desktop\\Dataloaders_tutorial\\frames"
    #frames_dir = extract_frames(video_path)

    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL image
        transforms.Resize((416, 416)),  # Resize to 416x416
        transforms.ToTensor(),
    ])

    dataset = FrameDataset(frames_dir,transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) 
    
    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame(columns=['Frame', 'Object', 'Class', 'Confidence', 'x_min', 'y_min', 'x_max', 'y_max'])

    frame_number = 0

    for batch_idx, batch in enumerate(data_loader):        
        results = model(batch)

        # Iterate through the results tensor
        for batch_idx in range(results.size(0)):  # Iterate over the batch
            
            result_df = extract_batch_results(result_df,results,batch_idx,frame_number)

            # Increment the frame number for the next frame
            frame_number += 1

    # Set the DataFrame index with two levels: 'Frame' and 'Object'
    result_df.set_index(['Frame', 'Object'], inplace=True)
    
    # calculate average num of persons in the video:
    average_num_of_persons = result_df.groupby(level=0).size().mean()

    return average_num_of_persons

if __name__ == "__main__":
    
    video_path = "data/bus.mp4"
    batch_size = 4 
    average_num_of_persons = process_video(video_path,batch_size)
    print(f"Average number of persons in video: {average_num_of_persons}")