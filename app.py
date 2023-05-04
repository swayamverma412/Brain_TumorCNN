#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from PIL import Image
import tensorflow as tf


# In[2]:


model = tf.keras.models.load_model('brain_tumor_model.h5')


# In[3]:


def predict(image):
    img = Image.open(image)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img/255.0
    prediction = model.predict(img)
    return prediction


# In[4]:


st.title("Brain Tumor Detection")

st.write("Upload an image of a brain scan to detect if it contains a tumor.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")


# In[6]:


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    prediction = predict(uploaded_file)
    if prediction > 0.5:
        st.write("The image contains a tumor with a probability of ", prediction[0][0])
    else:
        st.write("The image does not contain a tumor with a probability of ", 1-prediction[0][0])


# In[ ]:




