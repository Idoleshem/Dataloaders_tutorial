
import os
import warnings
import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.ops import nms

from helpers import extract_batch_results, extract_frames

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

def process_video(video_path,batch_size):

    # parameters and path
    num_of_frames = 16
    frames_dir = "C:\\Users\\Ido\\Desktop\\Dataloaders_tutorial\\frames"

    # extract frames
    frames_dir = extract_frames(video_path,num_of_frames)

    # load model + evaluate mode
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.eval()

    # define transformation
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL image
        transforms.Resize((640, 640)),  # Resize
        transforms.ToTensor(),
    ])

    # define the dataset ad dataloader
    dataset = FrameDataset(frames_dir,transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False) 
    
    # Create an empty DataFrame to store the results
    result_df = pd.DataFrame(columns=['Frame', 'Object', 'Class', 'Confidence', 'x_min', 'y_min', 'x_max', 'y_max'])

    frame_number = 0 # counter

    for batch_idx, batch in enumerate(data_loader):        
        
        # infer on batch
        results = model(batch) # result is a tensor ([batch_idx,channels,width, height ])

        # Iterate through the batch results tensor
        for frame_idx_in_batch in range(results.size(0)):  # Iterate over the batch
            
            result_df = extract_batch_results(result_df,results,frame_idx_in_batch,frame_number)

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


