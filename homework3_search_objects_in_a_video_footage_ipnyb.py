import streamlit as st
import tensorflow as tf
import cv2
import os
import shutil
from tensorflow.keras.applications import InceptionV3
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

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

# Function to detect objects in the frames
def detect_all_objects(frame_filenames):
    if not frame_filenames:
        st.warning("No frames found")
        return None, None
    else:
        model = InceptionV3(weights='imagenet', include_top=True)
        frame_obj_dict = {}
        for frame_filename in frame_filenames:
            img = tf.io.read_file(frame_filename)
            img = tf.image.decode_image(img, channels=3)
            img = tf.cast(img, tf.float32)
            img = tf.image.resize(img, (299, 299))
            img = tf.keras.applications.inception_v3.preprocess_input(img)
            img_expanded = tf.expand_dims(img, axis=0)
            prediction = model.predict(img_expanded, verbose=0)
            decoded_predictions = tf.keras.applications.inception_v3.decode_predictions(prediction, top=3)
            img_dict = {}
            for i, (imagenet_id, label, score) in enumerate(decoded_predictions[0]):
                img_dict[i + 1] = label
            frame_obj_dict[frame_filename] = img_dict

        all_objects = []
        for frame_name in frame_obj_dict.keys():
            for obj_id in frame_obj_dict[frame_name].keys():
                all_objects.append(frame_obj_dict[frame_name][obj_id])
        all_obj = list(set(all_objects))

        return frame_obj_dict, all_obj

# Function to search for a specific object in the frames
def search_object(search_query, frame_obj_dict, all_obj):
    if not frame_obj_dict:
        st.warning("No frames found")
        return None
    else:
        obj_frames = []
        for frame_name in frame_obj_dict.keys():
            if search_query in frame_obj_dict[frame_name].values():
                obj_frames.append(frame_name)

        if len(obj_frames) == 0:
            st.warning("Object doesn't exist!")
            st.write("Choose from the list below:")
            for obj in all_obj:
                st.write(obj)
        else:
            for framee in obj_frames:
                frame_path = framee
                frame = cv2.imread(frame_path)
                st.image(frame, caption=f"{search_query} in {framee}", use_column_width=True)

# Streamlit app layout
st.title("Object Detection in Video Frames")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
search_query = st.text_input("Enter the object to search for:")

if st.button("Search"):
    if uploaded_file and search_query:
        frame_filenames = get_frames(uploaded_file)
        if frame_filenames:
            frame_obj_dict, all_obj = detect_all_objects(frame_filenames)
            search_object(search_query, frame_obj_dict, all_obj)
    elif not uploaded_file:
        st.warning("Please upload a video file.")
    elif not search_query:
        st.warning("Please enter an object to search for.")
