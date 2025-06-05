import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import requests
from io import BytesIO
from PIL import Image

img_height = 180
img_width = 180

model = load_model('D:\Projects\AI Projects\Fruit_Images_Classification_Model\Image_classify.keras')
st.header('Image Classification Model')
img = st.text_input('Enter your image name: ','Garlic.jpg')

data_categories = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']

image_load = tf.keras.utils.load_img(img, target_size=(img_height,img_width))
image_array = tf.keras.utils.img_to_array(image_load)
image_bat = tf.expand_dims(image_array,0)

predict = model.predict(image_bat)

score = tf.nn.softmax(predict)
st.image(img)
st.write('The entered image is a {} with accuracy of {:0.2f}'.format(data_categories[np.argmax(score)],np.max(score)*100))