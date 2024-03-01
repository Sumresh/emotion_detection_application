import cv2
from PIL import Image
from transformers import pipeline
import streamlit as st

st.title("Live Emotion")

# Load the pre-trained face detection classifier
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize the image classification pipeline
classifier = pipeline("image-classification", model="trpakov/vit-face-expression")
video = cv2.VideoCapture(0)
# Function to process video frames
def main():
    
    FRAME_WINDOW = st.image([])
    i = 0
    if st.button("Stop", key=i):
        video.release()
        cv2.destroyAllWindows()
        return
    
    while True:
        i += 1
        # Read a frame from the webcam
        ret, frame = video.read()
        if not ret:
            break
        
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces in the frame
        faces = facedetect.detectMultiScale(frame_rgb, 1.3, 5)
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the cropped face region
            face_img = frame_rgb[y:y+h, x:x+w]
            
            # Resize the face image to the required input size of the classification model
            resized_img = cv2.resize(face_img, (224, 224))
            
            # Convert the NumPy array to a PIL image object
            pil_image = Image.fromarray(resized_img)
            
            # Make prediction using the image classification pipeline
            output = classifier(pil_image)
            
            # Extract the predicted emotion
            predicted_emotion = output[0]['label']
            
            # Display the predicted emotion on the frame
            cv2.putText(frame_rgb, predicted_emotion, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Draw a rectangle around the detected face
            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (50, 50, 250), 2)
        
        # Display the frame
        FRAME_WINDOW.image(frame_rgb)
        
        # Check if the user clicked the "Stop" button
        

    # Release the video capture object and close all windows

# Start the main function
if st.button("start"):
    main()