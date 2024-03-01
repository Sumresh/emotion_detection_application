import cv2
import os
import random
from transformers import pipeline
from PIL import Image
import os
from collections import Counter
from collections import defaultdict
import shutil
import streamlit as st
import pandas as pd

uploaded_file = st.file_uploader("Choose a file", type=["avi", "mp4"])
if uploaded_file is not None:
    # def extract_frames(video_path, output_folder):
        # Open the video file
        save_directory = os.getcwd()
        os.makedirs(save_directory, exist_ok=True)
        filename = uploaded_file.name
        save_path = os.path.join(save_directory, "video.mp4")
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"File saved successfully")
        video_capture = cv2.VideoCapture("video.mp4")

        # Read the video frame by frame
        success, frame = video_capture.read()
        count = 0

        folder_path = "frames"

    # Use os.makedirs() to create the folder and any necessary parent folders
        os.makedirs(folder_path, exist_ok=True)
        # Loop through each frame and save it as an image
        while success:
            # Save frame as JPEG file
            frame_path = f"{folder_path}/frame_{count}.jpg"
            cv2.imwrite(frame_path, frame)

            # Read the next frame
            success, frame = video_capture.read()
            count += 1

        # Release the video capture object
        video_capture.release()

    # Example usage
    # video_path = "C:\Users\abhil\Downloads\cyber_hackathon1\cyber_hackathon\flasktry\sample.mp4"
    # video_path = r"C:\Users\abhil\Downloads\cyber_hackathon1\cyber_hackathon\flasktry\sample.mp4"
        # video_path = "sample.mp4"


        # output_folder = "./frames"
        # extract_frames(video_path, output_folder)




        frames_directory = "frames"
        selected_frames_directory = "selected_frames"

        # Create a directory to store selected frames
        os.makedirs(selected_frames_directory, exist_ok=True)

        # Get a list of all frame filenames in the directory
        frame_files = os.listdir(frames_directory)

        # Calculate the number of frames to select (one-tenth of the total frames)
        num_frames_to_select = len(frame_files) // 15

        # Select random frames
        random_frames = random.sample(frame_files, num_frames_to_select)

    # Print the randomly selected frame filenames
    # print("Randomly selected frames:")
    # for frame in random_frames:
    #     print(frame)

    # Move selected frames to the new directory and delete the rest
        for frame in frame_files:
            frame_path = os.path.join(frames_directory, frame)
            if frame in random_frames:
                shutil.move(frame_path, os.path.join(selected_frames_directory, frame))
            else:
                os.remove(frame_path)

        # Remove the original frames directory
        os.rmdir(frames_directory)




        # Load the image classification pipeline
        classifier = pipeline("image-classification", model="trpakov/vit-face-expression")

        # Specify the directory containing the images
        images_directory = "selected_frames"

        # Get a list of all image filenames in the directory
        image_files = os.listdir(images_directory)
        emo=[]
        # Iterate over each image file
        for image_file in image_files:
            # Construct the full path to the image file
            image_path = os.path.join(images_directory, image_file)

            # Load the image
            image = Image.open(image_path)

            # Make prediction
            emotion_prediction = classifier(image)

            # Extract the predicted emotion
            predicted_emotion = emotion_prediction[0]

            # Print the predicted emotion for the current image
            emo.append(predicted_emotion)
            print(f"Predicted emotion for {image_file}: {predicted_emotion}")

        # print(emo)

        label_scores = defaultdict(lambda: {'sum': 0, 'count': 0})

        # Iterate through the data and update the sum and count of scores for each label
        for item in emo:
            label = item['label']
            score = item['score']
            label_scores[label]['sum'] += score
            label_scores[label]['count'] += 1

        # Calculate the average score for each label
        average_scores = {label: label_info['sum'] / label_info['count'] for label, label_info in label_scores.items()}

        # Print the dictionary of label and average score for each type of label
        for label, avg_score in average_scores.items():
            print(f"{label}: {avg_score:.2f}")
            output1= "{:.2f}".format(avg_score)

            st.write(f"{label}: {avg_score:.2f}")

        labels = list(average_scores.keys())
        avg_scores = list(average_scores.values())

# # Create bar chart
        # st.bar_chart(y=avg_scores, x=labels)
        st.bar_chart(average_scores, use_container_width=True)

        # data=average_scores

        # df = pd.DataFrame(data)


        # df_sorted = df.sort_values(by='avg_score', ascending=False)

        # st.bar_chart(df.set_index('label')['avg_score'], use_container_width=True, color="#87CEEB")










# import streamlit as st
# import cv2
# import numpy as np
# import os

# def main():
#     st.title("Video Recorder in Streamlit")

#     # Create a button to start recording
#     record_button = st.button("Start Recording")

#     # Initialize video capture from webcam
#     cap = cv2.VideoCapture(0)

#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

#     if record_button:
#         # Start recording
#         st.write("Recording started...")
#         recording = True

#         # Display video frames and start recording
#         while recording:
#             ret, frame = cap.read()

#             if not ret:
#                 st.error("Failed to capture video frame")
#                 break

#             # Convert frame to RGB format
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#             # Display the frame in Streamlit
#             st.image(rgb_frame, channels="RGB")

#             # Generate a unique key for the stop button
#             stop_button_key = f"stop_button_{record_button}"

#             # Check if the stop recording button is clicked
#             if st.button("Stop Recording", key=stop_button_key):
#                 recording = False

#             # Write the frame to the output file
#             out.write(frame)

#         st.write("Recording stopped")

#     # Release the video capture and writer objects
#     cap.release()
#     out.release()

# if __name__ == "__main__":
#     main()
