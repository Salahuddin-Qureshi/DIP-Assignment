import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Function to display images
def show_images(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    st.pyplot(fig)

# 1. Gamma Correction
def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

# 2. Histogram Equalization
def histogram_equalization(image):
    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

# 3. Laplacian Filtering
def laplacian_filter(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.convertScaleAbs(laplacian)

# 4. Sobel Operator
def sobel_operator(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    return cv2.convertScaleAbs(sobel_x + sobel_y)

# 5. Lowpass and Highpass Filters
def lowpass_filter(image):
    kernel = np.ones((5, 5), np.float32) / 25
    return cv2.filter2D(image, -1, kernel)

def highpass_filter(image):
    lowpass = lowpass_filter(image)
    return cv2.subtract(image, lowpass)

# Main Streamlit app
def main():
    st.title("Image Enhancement Techniques")

    # Upload images
    gamma_image_file = st.file_uploader("Upload image for Gamma Correction", type=["jpg", "png", "jpeg"])
    hist_eq_image_file = st.file_uploader("Upload image for Histogram Equalization", type=["jpg", "png", "jpeg"])
    laplacian_image_file = st.file_uploader("Upload image for Laplacian Filtering", type=["jpg", "png", "jpeg"])
    sobel_image_file = st.file_uploader("Upload image for Sobel Operator", type=["jpg", "png", "jpeg"])
    lowpass_image_file = st.file_uploader("Upload image for Lowpass Filter", type=["jpg", "png", "jpeg"])
    highpass_image_file = st.file_uploader("Upload image for Highpass Filter", type=["jpg", "png", "jpeg"])

    # Process each uploaded image
    if gamma_image_file:
        gamma_image = Image.open(gamma_image_file)
        gamma_image = cv2.cvtColor(np.array(gamma_image), cv2.COLOR_RGB2BGR)
        gamma_img = gamma_correction(gamma_image, 2.0)

        # Display results
        st.subheader("Gamma Correction")
        show_images([gamma_image, gamma_img], ['Original', 'Gamma Corrected'])

    if hist_eq_image_file:
        hist_eq_image = Image.open(hist_eq_image_file)
        hist_eq_image = cv2.cvtColor(np.array(hist_eq_image), cv2.COLOR_RGB2BGR)
        hist_eq_img = histogram_equalization(hist_eq_image)

        # Display results
        st.subheader("Histogram Equalization")
        show_images([hist_eq_image, hist_eq_img], ['Original', 'Histogram Equalized'])

    if laplacian_image_file:
        laplacian_image = Image.open(laplacian_image_file)
        laplacian_image = cv2.cvtColor(np.array(laplacian_image), cv2.COLOR_RGB2BGR)
        laplacian_img = laplacian_filter(laplacian_image)

 # Display results
        st.subheader("Laplacian Filtering")
        show_images([laplacian_image, laplacian_img], ['Original', 'Laplacian Filtered'])

    if sobel_image_file:
        sobel_image = Image.open(sobel_image_file)
        sobel_image = cv2.cvtColor(np.array(sobel_image), cv2.COLOR_RGB2BGR)
        sobel_img = sobel_operator(sobel_image)

        # Display results
        st.subheader("Sobel Operator")
        show_images([sobel_image, sobel_img], ['Original', 'Sobel Filtered'])

    if lowpass_image_file:
        lowpass_image = Image.open(lowpass_image_file)
        lowpass_image = cv2.cvtColor(np.array(lowpass_image), cv2.COLOR_RGB2BGR)
        lowpass_img = lowpass_filter(lowpass_image)

        # Display results
        st.subheader("Lowpass Filter")
        show_images([lowpass_image, lowpass_img], ['Original', 'Lowpass Filtered'])

    if highpass_image_file:
        highpass_image = Image.open(highpass_image_file)
        highpass_image = cv2.cvtColor(np.array(highpass_image), cv2.COLOR_RGB2BGR)
        highpass_img = highpass_filter(highpass_image)

        # Display results
        st.subheader("Highpass Filter")
        show_images([highpass_image, highpass_img], ['Original', 'Highpass Filtered'])

if __name__ == "__main__":
    main()