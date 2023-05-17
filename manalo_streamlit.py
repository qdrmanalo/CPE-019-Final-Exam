#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import tensorflow as tf

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('sky_classification_model.hdf5')
  return model
model=load_model()
st.write("""
# Sky Classification System"""
)
file=st.file_uploader("Choose sky photo from computer",type=["jpg","png"])

from PIL import Image,ImageOps
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import cv2
def import_and_predict(image_data,model):
    input_arr = img_to_array(image_data)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    prediction = model.predict(input_arr)
    return prediction
 
if file is None:
    st.text("Please upload an image file")
else:
    st.image(Image.open(file),use_column_width=True)
    image=load_img(file, target_size=(32,32,3))
    prediction=import_and_predict(image,model)
    class_names=['Cloudy', 
                 'Rain', 
                 'Sunrise', 
                 'Shine']
    string="OUTPUT : "+class_names[np.argmax(prediction)],prediction
    st.success(string)

