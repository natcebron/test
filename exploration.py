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
import cv2 as cv


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  
    local_css(os.path.join(currentdir, "style.css"))
    #Préparation de la page
    st.markdown(""" <style> .font {font-size:20px ; font-family: 'Arial'; color: #FFFFFF;} </style> """, unsafe_allow_html=True)
    st.markdown("# DATA EXPLORATION")
    st.markdown('<p class="font"> This part aims to study the characteristics of the analyzed dataset (number of images, proportion per group, visualization). </p>', unsafe_allow_html=True)

    st.markdown('## Dataset Design')

    comptage = pd.read_csv(os.path.join(currentdir, 'data/comptage.csv'),sep=';')

    fig = plt.figure(figsize=(15,10),facecolor='#0e1117')
    plt.rcParams.update({'text.color': "white",'axes.labelcolor': "white",'text.color':'white','xtick.color':'white','ytick.color':'white','axes.facecolor':'#0e1117','axes.edgecolor':'#0e1117'})
    plt.subplot(221)
    plt.barh(comptage.index,comptage['Nombre images'],color = ('red','green','blue'))
    plt.ylabel("Labels");
    plt.xlabel("Picture number");
    plt.title("Number of images for each condition");
    plt.subplot(222)
    plt.pie(comptage['percent'],autopct='%1.1f%%', labels = comptage['group'],colors = ('red','green','blue'));
    plt.title("Percentage for each condition");
    st.pyplot(fig)

    st.markdown('<p class="font"> Our dataset contains 3 sets of images, the first set contains 11263 images that belong to the Non_Covid condition. The second set contains 11957 images belonging to the COVID condition and the last set contains 10701 images corresponding to the normal condition. </p>', unsafe_allow_html=True)
    st.markdown('<p class="font"> The distribution of images between the sets is respected with approximately 33% per set. </p>', unsafe_allow_html=True)

    st.markdown('## X-rays Visualization')

    # Fonction pour charger l'image

    col1, col2,col3 = st.columns([1,1,1])
    with col1:
                covid = Image.open('data/covid_1.png')
                st.image(covid,width=300,use_column_width='never',caption='COVID')
    with col2:
                normal = Image.open('data/Normal (1).png')
                st.image(normal,width=300,use_column_width='never',caption='Normal')
    with col3:
                n_covid = Image.open('data/non_COVID (1).png')
                st.image(n_covid,width=300,use_column_width='never',caption='Non COVID')


    st.markdown('## X-rays Dimensions')

    image = Image.open(os.path.join(currentdir, 'data/Presentation1.png'))
   
    col1, mid, col2 = st.columns([2,2,2])
    with mid:
        st.image(image, width=300)

    st.markdown('<p class="font"> The images present in our dataset are all of dimension 256 pixels in height and 256 pixels in width. </p>', unsafe_allow_html=True)

    return None
