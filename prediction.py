import streamlit as st
import streamlit.components.v1 as components
import os                      #+Deployment
import inspect                 #+Deployment
#importing all the necessary libraries
import pandas as pd
import numpy as np                     
import matplotlib.pyplot as plt
import os
import random
import seaborn as sns
from PIL import Image, ImageStat
import matplotlib.image as mpimg
import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models,utils
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.python.keras import utils
import plotly.express as px


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  
    local_css(os.path.join(currentdir, "style.css"))
    #Préparation de la page
    st.markdown(""" <style> .font {font-size:20px ; font-family: 'Arial'; color: #FFFFFF;} </style> """, unsafe_allow_html=True)
    st.markdown("# X-RAYS PREDICTION")
    dog_breeds_category_path = os.path.join(currentdir, 'data/test.pkl')

    predictor_model = load_model(os.path.join(currentdir, 'model.hdf5'))

    with open(dog_breeds_category_path, 'rb') as handle:
        dog_breeds = pickle.load(handle)
    #importing all the helper fxn from helper.py which we will create later
    result = []
    def predictor(img_path): # here image is file name 
        image = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to a batch.
        input_arr = input_arr.astype('float32') / 255.  # This is VERY important
        predictions = predictor_model.predict(input_arr)
        lst2 = ['COVID', 'Non_COVID','Normal']
        test = pd.DataFrame(np.round(predictions,2),columns = lst2).transpose()
        test.columns = ['values']
        test = test.reset_index()
        test.columns = ['name', 'values']
  

        return test
    


    sns.set_theme(style="darkgrid")

    sns.set()

    from PIL import Image


    def save_uploaded_file(uploaded_file):
        try:
            with open(os.path.join('data/images',uploaded_file.name),'wb') as f:
                f.write(uploaded_file.getbuffer())
            return 1    
        except:
            return 0

    uploaded_file = st.file_uploader("Upload Image")

# text over upload button "Upload Image"

    if uploaded_file is not None:
        if save_uploaded_file(uploaded_file): 
        # display the image
            display_image = Image.open(uploaded_file)
            col1, col2 = st.columns([1,1])
            with col1:
                st.image(display_image,width=400,use_column_width='never')
            prediction = predictor(os.path.join('data/images',uploaded_file.name))
            prediction = prediction.round(decimals = 2)
            os.remove('data/images/'+uploaded_file.name)
            with col2:
                st.image(display_image,width=400,use_column_width='never')
            fig = px.bar(prediction,x = "values",y = "name",title = "Prediction result",color="name",orientation = 'h',text='values')
            fig.update_layout(width=900,height=600)

            st.plotly_chart(fig)
            
         

    
