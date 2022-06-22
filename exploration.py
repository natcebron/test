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
    st.markdown("# EXPLORATION DE DONNÉES")
    st.markdown("<p class='font'> Cette partie vise à étudier les caractéristiques du jeu de données analysé (nombre d'images, proportion par groupe, visualisation). </p>", unsafe_allow_html=True)

    st.markdown("## Conception de l'ensemble de données")

    comptage = pd.read_csv(os.path.join(currentdir, 'data/comptage.csv'),sep=';')

    fig = plt.figure(figsize=(15,10),facecolor='#0e1117')
    plt.rcParams.update({'text.color': "white",'axes.labelcolor': "white",'text.color':'white','xtick.color':'white','ytick.color':'white','axes.facecolor':'#0e1117','axes.edgecolor':'#0e1117'})
    plt.subplot(221)
    plt.barh(comptage.index,comptage['Nombre images'],color = ('red','green','blue'))
    plt.ylabel("Labels");
    plt.xlabel("Numéro d'image");
    plt.title("Nombre d'images pour chaque condition");
    plt.subplot(222)
    plt.pie(comptage['percent'],autopct='%1.1f%%', labels = comptage['group'],colors = ('red','green','blue'));
    plt.title("Pourcentage pour chaque condition");
    st.pyplot(fig)

    st.markdown("<p class='font'> Notre jeu de données contient 3 ensembles d'images, le premier ensemble contient 11263 images qui appartiennent à la condition Non_Covid. Le deuxième ensemble contient 11957 images appartenant à la condition COVID et le dernier ensemble contient 10701 images correspondant à la condition normale. </p>", unsafe_allow_html=True)
    st.markdown("<p class='font'> La répartition des images entre les ensembles est respectée avec environ 33% par ensemble. </p>", unsafe_allow_html=True)

    st.markdown('## Visualisation de radiographies')

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


    st.markdown('## Dimensions des radiographies')

    image = Image.open(os.path.join(currentdir, 'data/Presentation1.png'))
   
    col1, mid, col2 = st.columns([2,2,2])
    with mid:
        st.image(image, width=300)

    st.markdown("<p class='font'> Les images présentes dans notre jeu de données sont toutes de dimension 256 pixels en hauteur et 256 pixels en largeur. </p>", unsafe_allow_html=True)

    return None
