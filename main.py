import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image

# Function to compute color histogram
def compute_color_histogram(image_path, color_space='LAB', bins=64):
    if not os.path.isfile(image_path):
        return None
    image = cv2.imread(image_path)
    if image is None:
        return None
   
    if color_space == 'HSV':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == 'LAB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    histograms = []
    for channel in range(image.shape[2]):
        hist = cv2.calcHist([image], [channel], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist)
        histograms.append(hist)
    return histograms

# Compare histograms using a similarity metric
def compare_histograms(hist1, hist2, method=cv2.HISTCMP_CHISQR):
    similarity = 0
    for channel in range(len(hist1)):
        # Normalize histograms
        hist1[channel] = cv2.normalize(hist1[channel], hist1[channel]).flatten()
        hist2[channel] = cv2.normalize(hist2[channel], hist2[channel]).flatten()
        
        # Calculate similarity
        score = cv2.compareHist(hist1[channel], hist2[channel], method)
        
        # Check for division by zero
        if score == 0:
            similarity_channel = 1  # or any other default value that makes sense for your use case
        else:
            similarity_channel = 1 / (1 + score)
        
        similarity += similarity_channel
        
    return similarity / len(hist1)

# Preprocess dataset and compute histograms
def process_jewellery_dataset(dataset_folder, color_space='LAB', bins=64):
    histograms = {}
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    for image_filename in os.listdir(dataset_folder):
        ext = os.path.splitext(image_filename)[1].lower()
        if ext in valid_extensions:
            image_path = os.path.join(dataset_folder, image_filename)
            hist = compute_color_histogram(image_path, color_space, bins)
            if hist is not None:
                histograms[image_filename] = hist
    return histograms

# Find similar images
def find_similar_images(query_image_path, dataset_histograms, color_space='LAB', bins=64, method=cv2.HISTCMP_CHISQR):
    query_hist = compute_color_histogram(query_image_path, color_space, bins)
    similarity_scores = {}
    for image_filename, dataset_hist in dataset_histograms.items():
        score = compare_histograms(query_hist, dataset_hist, method)
        similarity_scores[image_filename] = score
    sorted_images = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_images

# Display the top similar images
def display_similar_images(query_image_path, similar_images, dataset_folder, top_n=5, columns = 2):
    st.write("### Query Image:")
    query_image = cv2.imread(query_image_path)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    st.image(query_image, caption="Query Image", width=200)

    st.write('### Top similar images:')

    # Use st.columns() to create columns for a better layout
    cols = st.columns(columns)  # Create 'columns' number of columns

    for i, (image_filename, score) in enumerate(similar_images[:top_n]):
        image_path = os.path.join(dataset_folder, image_filename)
        similar_image = cv2.imread(image_path)
        similar_image = cv2.cvtColor(similar_image, cv2.COLOR_BGR2RGB)
        
        # Display each image in one of the columns
        with cols[i % columns]:  # Distribute images across the columns
            st.image(similar_image, caption=f'{image_filename} - Similarity: {score:.2f}', width=200)  # Resized similar image width

# Ensure the 'tmp' directory exists
if not os.path.exists('tmp'):
    os.makedirs('tmp')

# Streamlit UI
st.title('Jewellery Image Similarity Search')

# Dataset selection
dataset_folder = 'training'
classes = [d for d in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, d))]
selected_class = st.selectbox("Select a Class", classes)

# Image upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Ensure image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    query_image = Image.open(uploaded_file)
    st.image(query_image, caption='Uploaded Image', use_column_width=True)

    # Preprocess dataset
    dataset_class_folder = os.path.join(dataset_folder, selected_class)
    st.write(f"Processing {selected_class} dataset...")
    dataset_histograms = process_jewellery_dataset(dataset_class_folder)

    # Save uploaded file to a temporary path
    query_image_path = f'tmp/{uploaded_file.name}'
    with open(query_image_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    # Find similar images
    similar_images = find_similar_images(query_image_path, dataset_histograms)

    # Display similar images
    display_similar_images(query_image_path, similar_images, dataset_class_folder)
