import streamlit as st
import tensorflow as tf
import cv2
import os
import shutil
import numpy as np
from PIL import Image
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image as keras_image

# Load the InceptionV3 model pre-trained on ImageNet
model = InceptionV3(weights='imagenet')

# Function to extract frames from the uploaded video
def get_frames(uploaded_file):
    if not uploaded_file:
        st.warning("No file uploaded")
        return None
    else:
        filename = uploaded_file.name
        with open(filename, mode='wb') as f:
            f.write(uploaded_file.getbuffer())
        
        if os.path.exists('frames'):
            shutil.rmtree('frames')
        os.makedirs('frames', exist_ok=True)
        
        cap = cv2.VideoCapture(filename)
        frame_count = 0
        frame_filenames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            frame_filename = f'frames/frame_{frame_count}.jpg'
            cv2.imwrite(frame_filename, frame)
            frame_filenames.append(frame_filename)
        cap.release()
        if frame_count == 0:
            st.warning("No frames extracted. Please check the video file.")
        else:
            st.success(f"Extracted {frame_count} frames.")
        return frame_filenames

# Function to detect objects in the frames and draw bounding boxes
def detect_objects(frame_filenames):
    frame_obj_dict = {}
    all_objects = []

    for frame_filename in frame_filenames:
        frame = cv2.imread(frame_filename)
        h, w = frame.shape[:2]

        # Preprocess the frame for InceptionV3
        img = keras_image.load_img(frame_filename, target_size=(299, 299))
        img_array = keras_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)

        # Predict using the InceptionV3 model
        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.inception_v3.decode_predictions(predictions, top=3)[0]

        objects = []
        for (imagenet_id, label, score) in decoded_predictions:
            objects.append({'label': label, 'score': score})
            all_objects.append(label)

        frame_obj_dict[frame_filename] = objects

        for obj in objects:
            label = obj['label']
            score = obj['score']
            # Draw bounding box (for demo purposes, drawing a fixed box)
            startX, startY, endX, endY = 50, 50, w-50, h-50
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, f"{label}: {score:.2f}", (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(frame_filename, frame)

    all_objects = list(set(all_objects))
    return frame_obj_dict, all_objects

# Function to search for a specific object in the frames
def search_object(search_query, frame_obj_dict):
    obj_frames = []
    for frame_name, objects in frame_obj_dict.items():
        if any(obj['label'] == search_query for obj in objects):
            obj_frames.append(frame_name)

    if len(obj_frames) == 0:
        st.warning("Object doesn't exist!")
    else:
        for framee in obj_frames:
            frame = cv2.imread(framee)
            st.image(frame, caption=f"{search_query} in {framee}", use_column_width=True)

# Streamlit app layout
st.title("Object Detection in Video Frames")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
search_query = st.text_input("Enter the object to search for:")

if st.button("Search"):
    if uploaded_file and search_query:
        frame_filenames = get_frames(uploaded_file)
        if frame_filenames:
            frame_obj_dict, all_obj = detect_objects(frame_filenames)
            search_object(search_query, frame_obj_dict)
    elif not uploaded_file:
        st.warning("Please upload a video file.")
    elif not search_query:
        st.warning("Please enter an object to search for.")
