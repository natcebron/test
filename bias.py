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


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  
    local_css(os.path.join(currentdir, "style.css"))
    #Préparation de la page
    st.markdown(""" <style> .font {font-size:20px ; font-family: 'Arial'; color: #FFFFFF;} </style> """, unsafe_allow_html=True)
    st.markdown("# BIAS DETERMINATION")

    st.markdown('## FIRST RESULT')
    st.markdown('<p class="font"> This section is dedicated to the determination of bias. In a first step, a model based on transfer learning was tested on the raw radiographs. We obtain the following results:  </p>', unsafe_allow_html=True)

    # initialize list of lists
    data = [['COVID', 0.99,0.90,0.94], ['Normal',0.89,0.92,0.90], ['Non_COVID',0.87,0.92,0.89],['Average',0.91,0.91,0.91]]
 
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=['Group', 'Precision','Recall','F1-score'])
    


    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.dataframe(data=df)

    with col3:
        st.write(' ')

    st.markdown('<p class="font"> The results obtained in this first model are very satisfactory, particularly for the COVID group, with precision values of 0.99. To check that the model is not biased, we performed a Grad-CAM study on the true positive and false negative images for each group.  </p>', unsafe_allow_html=True)

    st.markdown('## GRAD-CAM')
    image = Image.open(os.path.join(currentdir, 'data/Gradcam.png'))
    new_image = image.resize((1400, 1000))

    st.markdown('<p class="font"> Grad-Cam is a method published in 2016 which aims to find out which parts of the image have been used by the model to classify the images. The result is displayed as a heatmap.  </p>', unsafe_allow_html=True)

    st.image(image,width=1200)
    st.markdown('<p class="font"> Using this method we obtain very conclusive results that our model is biased. Indeed, we can see that the areas mostly used by the model (yellow) do not correspond to the lungs but to external parts of the image. This result is even more obvious for the COVID group where we can see that these are always the same areas found. This result explains our accuracy for this very high group.   </p>', unsafe_allow_html=True)


    st.markdown('## BIASED DATASET')

    st.markdown('## GANS')


