import pandas as pd
import streamlit as st
import cv2 
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
fig = plt.figure()


################ Starting of CSS Module ############################
with open(r"~/styles/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

################ Ending of CSS Module ############################

############### Starting of TCS design (in-dev) ################

st.title('TCS Smart Retail Store')

st.sidebar.write("Use sidebar to add products to cart.")

################### End #####################################

################### Start of TCS drop down sidebar ########################

st.sidebar.write("Select product code from drop down menu")

options = {
    "None":"Select Option",
    "Oreo":"Oreo",
    "Red Label Tea":"Red Label Tea",
    "Bru":"Bru",
    "Maggi":"Maggi"
}

option = st.sidebar.selectbox("Product details", list(options.items()), 0 , format_func=lambda o: o[1])
st.sidebar.write("Your selection is: ")
st.sidebar.write(option[0])

################### End of TCS drop down sidebar ########################

########### Prediction Function ###################################
img_height = 350
img_width = 350
def predict(image):
	#Upload the model saved using the 4 products data augmented section
    modelsaved = r"~/models/retail_model_augmented_05272021_183451.h5"
    IMAGE_SHAPE = (530, 530,3)
    model = load_model(modelsaved, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    img = keras.preprocessing.image.load_img(r"~/images/CapturedImage.jpg",target_size=(img_height, img_width))

##### Class_names may need to be modified below and in dropdown box.
##### Model has to be changed
    class_names = ['Bru','Maggi','Oreo','Red Label Tea']
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    scores = tf.nn.softmax(predictions[0])
    print(
    "This image is {} "
    .format(class_names[np.argmax(scores)])
    )
    scores = scores.numpy()
    result = f"{class_names[np.argmax(scores)]}" 
    st.write(result)
    if (result == option[0]):
       {
	st.success("success")
       }
    else:
       {
        st.error("Incorrect product detected. Transaction has been flagged for manual verification")
       }


########### Prediction Function ends###################################

########## Capturing the image using webcam  #########################
MAX_FRAMES = 30
#st.slider("Select number of frames to render:", min_value=60, max_value=600, value=300, step=30)
run = st.sidebar.button("Verify Product")

if run:
 st.error("Warning Camera about to activate. ALLOW CAMERA access in sidebar")
 
warn=st.sidebar.button("Allow Camera")
 
if warn:
  with st.spinner('Please wait while product is being scanned'):
 
        capture = cv2.VideoCapture(0)
        img_display = st.empty()
        for i in range(MAX_FRAMES):
                cap, frame = capture.read()
                cv2.imwrite("~/images/CapturedImage.jpg",frame)
                result = False

        image = Image.open("~/images/CapturedImage.jpg")
        image = image.convert('RGB')
        st.image(image, caption='Captured Image', use_column_width=True)
        plt.imshow(image)
        plt.axis("off")
        predictions = predict(image)
        time.sleep(1)
        #st.success('Success')
        st.write(predictions)
        st.markdown("Render complete")
        capture.release()

                
########### Ending the capture of image using webcam	 ################

########### Function to rerun the program  ######################
rerun = st.sidebar.button("Rerun")

if rerun:
 with st.spinner('Please hold while your application reruns'):
  time.sleep(4)
  st.experimental_rerun()
 
