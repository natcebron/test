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
from PIL import Image, ImageStat,ImageOps
import matplotlib.image as mpimg
import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models,utils
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.python.keras import utils
import keras
import matplotlib.cm as cm
import tensorflow

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  
    local_css(os.path.join(currentdir, "style.css"))
    #Préparation de la page
    st.markdown(""" <style> .font {font-size:20px ; font-family: 'Arial'; color: #FFFFFF;} </style> """, unsafe_allow_html=True)
    st.markdown("# PRE-PROCESSING METHODS")
    st.markdown('<p class="font"> In an attempt to eliminate bias, various pre-processing methods were tested. We tested the application of masks using a UNET model, gaussian filter: A method of blurring the image to reduce image noise, adjust gamma: A correction method for controlling brightness, contrast limited adaptive histogram equalization (CLAHE), a technique for modifying the image by improving the contrast. The last method is a mixture of CLAHE and a transformation filter. </p>', unsafe_allow_html=True)

    image = Image.open(os.path.join(currentdir, 'data/preprocessing method.png'))
    st.image(image,width=1200)
    st.markdown("# MODEL RESULTS")
    st.markdown('<p class="font">  Several application models for these transformations have been tested. The basic model used is Inceptionv3, which is a model widely used in the field of X-ray imaging.</p>', unsafe_allow_html=True)
    unet = load_model(os.path.join(currentdir, 'data/UNET.hdf5'))

    df = pd.read_csv(os.path.join(currentdir, 'data/model results.csv'),sep=";")
    st.dataframe(data=df)
    df2 = pd.read_csv(os.path.join(currentdir, 'data/model results2.csv'),sep=";")
    st.dataframe(data=df2)
    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index = None):
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()   

    model_builder = keras.applications.xception.Xception
    img_size = (299, 299)
    preprocess_input = keras.applications.xception.preprocess_input
    decode_predictions = keras.applications.xception.decode_predictions
    last_conv_layer_name = "block14_sepconv2_act"
    def Gradcam(url):
        img = tensorflow.keras.preprocessing.image.load_img(url, target_size = img_size) 
        array = tensorflow.keras.preprocessing.image.img_to_array(img) 
        array = np.expand_dims(array, axis = 0)
        model = model_builder(weights = "imagenet")
        model.layers[-1].activation = None
        preds = model.predict(array) 
        heatmap = make_gradcam_heatmap(array, model, last_conv_layer_name)
        img = tensorflow.keras.preprocessing.image.load_img(url)
        img = tensorflow.keras.preprocessing.image.img_to_array(img)
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = tensorflow.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = tensorflow.keras.preprocessing.image.img_to_array(jet_heatmap)
        superimposed_img = jet_heatmap * 1 + img
        superimposed_img = tensorflow.keras.preprocessing.image.array_to_img(superimposed_img)
        plt.axis('off')
        rouge, vert, bleu = superimposed_img.split()
        image_array = np.array(rouge,dtype='float64')
        cv2.imwrite('data/images/gradcam.png',image_array)
        return superimposed_img
    
    def m_unet(img):
        img = tf.keras.preprocessing.image.load_img(img, target_size=(256, 256,3))
        input_arr = tf.keras.preprocessing.image.img_to_array(img)
        img = cv2.cvtColor(input_arr, cv2.COLOR_BGR2GRAY)
        img2 = img /255.
        test_img_input=np.expand_dims(img2, axis=(0, 3))
        prediction = (unet.predict(test_img_input)[0,:,:,0] > 0.5).astype(np.uint8)
        z = (prediction * 255).astype(np.uint8)
        z2 = Image.fromarray((z).astype('uint8'), mode='L')
        z3 = Image.fromarray((img).astype('uint8'), mode='L')
        im_out = Image.composite(z3, z2, z2)  # apply mask
        img5 = np. array(im_out,dtype='float64')
        img5 = np.expand_dims(img5, axis=-1)
        img5.shape
        cv2.imwrite('data/images/savedImage.png',img5)
        
    st.markdown('<p class="font">  When we apply the mask we have a decrease in model performance for the benefit of interpretability. As far as image transformations are concerned, it is with the contrast strech that we obtain the best results.</p>', unsafe_allow_html=True)

    st.markdown("# GRADCAM")
    st.markdown('<p class="font">  Function allowing from an imported image to identify the regions used by the deep learning model.</p>', unsafe_allow_html=True)

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
            display_image = Image.open(uploaded_file).convert('L')
            array = tensorflow.keras.preprocessing.image.img_to_array(display_image) 
            col1, mid,col2 = st.columns([3,3,3])
            fig = plt.figure(figsize=(12, 12))
            plt.imshow(array, cmap='gray')
            plt.axis('off')

            with col1:
                st.header("Upload picture")
                st.pyplot(fig,use_column_width=True)
                
            fig2 = plt.figure(figsize=(12, 12))
            plt.imshow(Gradcam(os.path.join('data/images',uploaded_file.name)))
            with mid:
                st.header("Gradcam before correction")
                st.pyplot(fig2,use_column_width=True)
            fig3 = plt.figure(figsize=(12, 12))
            m_unet(os.path.join('data/images',uploaded_file.name))
            plt.imshow(Gradcam(os.path.join('data/images/savedImage.png')))
            
            with col2:
                st.header("Gradcam after correction")

                st.pyplot(fig3)


            
         

    
