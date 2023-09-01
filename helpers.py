import cv2
import os

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
