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
import streamlit.components.v1 as components
import plotly.express as px
import random 

def app():
    predictor_model = load_model(os.path.join('Model_masks.hdf5'))
    unet = load_model(os.path.join('UNET.hdf5'))
    def Gradcam(url):
        img = keras.preprocessing.image.load_img(url, target_size = img_size) 
        array = keras.preprocessing.image.img_to_array(img) 
        array = np.expand_dims(array, axis = 0)
        model = model_builder(weights = "imagenet")
        model.layers[-1].activation = None
        preds = model.predict(array) 
        heatmap = make_gradcam_heatmap(array, model, last_conv_layer_name)
        img = keras.preprocessing.image.load_img(url)
        img = keras.preprocessing.image.img_to_array(img)
        heatmap = np.uint8(255 * heatmap)
        jet = cm.get_cmap("jet")
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap]
        jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
        jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
        superimposed_img = jet_heatmap * 1 + img
        superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
        plt.axis('off')
        rouge, vert, bleu = superimposed_img.split()
        image_array = np.array(superimposed_img,dtype='float64')
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

    st.markdown("# PICTURE INTERACTIVE ANALYSIS V2")

    selected_box = st.selectbox(
    'Choose one of the following',
    ('COVID','Normal','Non Normal')
    )
   
    if selected_box == 'COVID':
        path="pictures/COVID"
        os.chdir("pictures/COVID")
        files=os.listdir(path)
        cov=random.choice(files)
        image2 = plt.imread(cov,format='png')

        col1, mid,col2 = st.columns([3,3,3])
        with col1:
            fig1 = plt.figure()
            plt.imshow(image2, cmap='gray')
            plt.axis('off')
            st.header("Upload picture")
            st.pyplot(fig1,use_column_width=True)

        with mid:
                fig2 = plt.figure(figsize=(12, 12))
                st.header("Gradcam before correction")
                plt.imshow(Gradcam(cov))
                st.pyplot(fig2)
 
        with col2:
                fig3 = plt.figure(figsize=(12, 12))
                
                m_unet(os.path.join(cov))
                st.header("Gradcam after correction")
                plt.imshow(Gradcam(os.path.join('data/images/savedImage.png')))
                st.pyplot(fig3)

        prediction = predictor(os.path.join('savedImage.png'))
        prediction = prediction.round(decimals = 2)
        fig = px.bar(prediction,x = "values",y = "name",title = "Prediction result",color="name",orientation = 'h',text='values')
        fig.update_layout(width=900,height=600)
        st.plotly_chart(fig)

    if selected_box == 'Normal':
        path="pictures/Normal"
        os.chdir("pictures/Normal")
        files=os.listdir(path)
        nor=random.choice(files)
        image3 = plt.imread(nor,format='png')

        col1, mid,col2 = st.columns([3,3,3])
        with col1:
            fig1 = plt.figure()
            plt.imshow(image3, cmap='gray')
            plt.axis('off')
            st.header("Upload picture")
            st.pyplot(fig1,use_column_width=True)

        with mid:
                fig2 = plt.figure(figsize=(12, 12))
                st.header("Gradcam before correction")
                plt.imshow(Gradcam(nor))
                st.pyplot(fig2)
 
        with col2:
                fig3 = plt.figure(figsize=(12, 12))
                
                m_unet(os.path.join(nor))
                st.header("Gradcam after correction")
                plt.imshow(Gradcam(os.path.join('data/images/savedImage.png')))
                st.pyplot(fig3)

        prediction = predictor(os.path.join('data/images/savedImage.png'))
        prediction = prediction.round(decimals = 2)
        fig = px.bar(prediction,x = "values",y = "name",title = "Prediction result",color="name",orientation = 'h',text='values')
        fig.update_layout(width=900,height=600)
        st.plotly_chart(fig)

    if selected_box == 'Non Normal':
        path="pictures/Non_COVID"
        os.chdir("pictures/Non_COVID")
        files=os.listdir(path)
        nnorm=random.choice(files)
        image4 = plt.imread(nnorm,format='png')

        col1, mid,col2 = st.columns([3,3,3])
        with col1:
            fig1 = plt.figure()
            plt.imshow(image4, cmap='gray')
            plt.axis('off')
            st.header("Upload picture")
            st.pyplot(fig1,use_column_width=True)

        with mid:
                fig2 = plt.figure(figsize=(12, 12))
                st.header("Gradcam before correction")
                plt.imshow(Gradcam(nnorm))
                st.pyplot(fig2)
 
        with col2:
                fig3 = plt.figure(figsize=(12, 12))
                
                m_unet(os.path.join(nnorm))
                st.header("Gradcam after correction")
                plt.imshow(Gradcam(os.path.join('data/images/savedImage.png')))
                st.pyplot(fig3)

        prediction = predictor(os.path.join('data/images/savedImage.png'))
        prediction = prediction.round(decimals = 2)
        fig = px.bar(prediction,x = "values",y = "name",title = "Prediction result",color="name",orientation = 'h',text='values')
        fig.update_layout(width=900,height=600)
        st.plotly_chart(fig)

    os.chdir("C:/Users/natha/DeMACIA-RX-main/Streamlit")

if __name__ == "__main__":
    app()
