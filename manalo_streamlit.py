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

import cv2
from PIL import Image,ImageOps
import numpy as np
def import_and_predict(image_data,model):
    size=(32,32)
    image=ImageOps.fit(image_data,size,Image.ANTIALIAS)
    img=np.asarray(image)
    img=cv2.resize(img,size,cv2.INTER_CUBIC)
    prediction=model.predict(img)
    return prediction
 
if file is None:
    st.text("Please upload an image file")
else:
    image=Image.open(file)
    st.image(image,use_column_width=True)
    prediction=import_and_predict(image,model)
    class_names=['Cloudy', 
                 'Rain', 
                 'Sunrise', 
                 'Shine']
    string="OUTPUT : "+class_names[np.argmax(prediction)]
    st.success(string)

