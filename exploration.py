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
    plt.pie(comptage['percent'],autopct='%1.1f%%', labels = comptage.index,colors = ('red','green','blue'));
    plt.title("Percentage for each condition");
    st.pyplot(fig)

    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    #First SubPlot
    fig = make_subplots(rows=1, cols=2)

    fig.add_trace(go.Pie(labels = comptage['group'],values = comptage['percent']), row=1, col=2)
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]), row=1, col=2)
    fig.update_layout(width=1500,height=400)
  
    st.plotly_chart(fig)


    st.markdown('<p class="font"> Our dataset contains 3 sets of images, the first set contains 11263 images that belong to the Non_Covid condition. The second set contains 11957 images belonging to the COVID condition and the last set contains 10701 images corresponding to the normal condition. </p>', unsafe_allow_html=True)
    st.markdown('<p class="font"> The distribution of images between the sets is respected with approximately 33% per set. </p>', unsafe_allow_html=True)

    st.markdown('## X-rays Visualization')

    # Fonction pour charger l'image

    # create figure
    fig = plt.figure(figsize=(20, 7))
    plt.rcParams.update({'text.color': "white",'axes.labelcolor': "white",'text.color':'white','xtick.color':'white','ytick.color':'white','axes.facecolor':'#0e1117','axes.edgecolor':'#0e1117'})
    fig.set_facecolor("#0e1117")
    # setting values to rows and column variables
    rows = 1
    columns = 3
  
    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)
  
    # showing image
    plt.imshow(cv.imread(os.path.join(currentdir, 'covid_1.png')), cmap='gray')
    plt.axis('off')
    plt.title("COVID",color= 'white',fontsize=15)
  
    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
  
    # showing image
    plt.imshow(cv.imread(os.path.join(currentdir, 'non_COVID (1).png')), cmap='gray')
    plt.axis('off')
    plt.title("Non COVID",color= 'white',fontsize=15)
  
    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)
  
    # showing image
    plt.imshow(cv.imread(os.path.join(currentdir, 'Normal (1).png')), cmap='gray')
    plt.axis('off')
    plt.title("Normal",color= 'white',fontsize=15);
    st.pyplot(fig)

    st.markdown('## X-rays Dimensions')

    image = Image.open(os.path.join(currentdir, 'data/Presentation1.png'))
   
    col1, mid, col2 = st.columns([2,2,2])
    with mid:
        st.image(image, width=512)

    st.markdown('<p class="font"> The images present in our dataset are all of dimension 256 pixels in height and 256 pixels in width. </p>', unsafe_allow_html=True)

    return None
