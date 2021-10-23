import keras
from numpy import asarray
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.write("""
Testing deepfake app
""")
model = keras.models.load_model('./model_140k.h5')

img = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if img is not None:
    image = Image.open(img)
    image = image.resize((64, 64), Image.ANTIALIAS)
    image = asarray(image)
    image = image / 255.0
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    predict = model.predict(image)
    p = np.argmax(predict)
    if (p == 1):
        st.success("REAL")
    else:
        st.error("FAKE")
