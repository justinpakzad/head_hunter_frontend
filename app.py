import streamlit as st
from PIL import Image, ImageDraw,ImageFont
import cv2
from roboflow import Roboflow
import io
import time
import numpy as np
font_path = 'Arial Black.ttf'
font = ImageFont.truetype(font_path, size=36)
url = 'https://detect.roboflow.com/crowd_counting/12'

# Hello World
rf = Roboflow(api_key=st.secrets['api_key'])
project = rf.workspace().project('crowd_counting')
model = project.version(14).model


def load_images(cv_image, confidence_threshold):
    # Make prediction using Roboflow API
    robo_prediction = model.predict(cv_image, confidence=confidence_threshold*100).json()
    st.markdown(f'<center><p style="font-size:25px; font-family: Sofia Pro">{len(robo_prediction["predictions"])} people detected</p></center>', unsafe_allow_html=True)
    pil_img_with_boxes = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img_with_boxes)
    img_width = cv_image.shape[0]
    for prediction in robo_prediction['predictions']:
        prediction['class'] = 'Person'
        label = prediction['class']
        xmin = prediction['x']
        ymin = prediction['y']
        width = prediction['width']
        height =  prediction['height']
        top_lx  = xmin  + prediction['width'] /2
        top_ly = ymin + prediction['height'] /2
        bottom_lx = xmin - prediction['width'] /2
        bottom_ly = ymin - prediction['height'] /2

        draw.rectangle(((top_lx, top_ly), (bottom_lx, bottom_ly)), outline='#00ff22', width=int(img_width  * 0.008))
        font = ImageFont.truetype(font_path, size=int(width * 0.3))
        draw.text((bottom_lx, bottom_ly), label, fill='white',font=font)
    st.image(pil_img_with_boxes)


def main():
    icon = Image.open('images/hh icon no bg.png')
    st.set_page_config(page_icon=icon,page_title='Head Hunter')
    image_logo_large = Image.open('images/hh logo no bg.png')
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(image_logo_large,width=700)
    with col2:
        st.write(' ')
    with col3:
        st.write(' ')
    with col4:
        st.write(' ')
    # st.markdown("<h1 style='text-align: center; font-family: Helvetica; color: #00ff22;'>Head Hunter</h1>", unsafe_allow_html=True)
    # image_logo = Image.open('/Users/justinpak/code/justinpakzad/head_hunter_frontend/hh icon no bg.png')
    # st.markdown('Counting crowds with confidence since 2023.')
    st.markdown("---")
    st.markdown("<h4 style='text-align: center; color: #ECB056; font-family: Sofia Pro ;'>Counting crowds with confidence since 2023</h4>", unsafe_allow_html=True)

# ECB056
    st.markdown("---")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.markdown("<h5 style='text-align: left; font-family: Sofia Pro; color: white;'>Upload a photo of your crowd here:</h5>", unsafe_allow_html=True)
    # st.write("Upload a photo of your crowd here")
    st.sidebar.header("Settings")

    # st.sidebar.header("Settings")
    page = st.sidebar.radio('',('Upload Photo','Take A Photo'))
    confidence_threshold = st.sidebar.slider('Confidence threshold:', 0.0, 1.0, 0.3, 0.01)
    # overlap_threshold = st.sidebar.slider('Overlap threshold:', 0.0, 1.0, 0.5, 0.01)
    if page == 'Upload Photo':
        img_file_buffer = st.file_uploader('')

        if img_file_buffer is not None:
            img_bytes = img_file_buffer.getvalue()
            # Load uploaded image
            pil_image = Image.open(io.BytesIO(img_bytes))
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            st.image(pil_image)
            if st.button("Predict"):
                with st.spinner('Hunting heads...'):
                    load_images(cv_image, confidence_threshold)

    if page == 'Take A Photo':
        img_file_buffer = st.camera_input('Say Cheese')

        if img_file_buffer is not None:
            img_bytes = img_file_buffer.getvalue()
            # Load uploaded image
            pil_image = Image.open(io.BytesIO(img_bytes))
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            # st.image(pil_image)
            if st.button("Predict..."):
                with st.spinner('Hunting heads...'):
                    load_images(cv_image, confidence_threshold, overlap_threshold)

if __name__ == '__main__':
    main()
