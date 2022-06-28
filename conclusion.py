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
    
    st.markdown(""" <style> .font {font-size:16px ; font-family: 'Arial'; color: #FFFFFF;} </style> """, unsafe_allow_html=True)
    st.markdown("# CONCLUSION")
    st.markdown("<p class='font'> Dans cette étude, la méthode d'interprétabilité du modèle d'apprentissage profond Gradcam, a été utilisée pour comprendre comment le modèle classe différents ensembles d'images. Cette méthode a confirmé l'hypothèse selon laquelle le jeu de données étudié était biaisé. En effet, nous avons montré que le modèle sur les images de base classait les images utilisant des parties non ciblées sur les poumons. Ce résultat a été confirmé en utilisant d'autres méthodes (jeu de données biaisé et réseaux adversariaux génératifs). L'utilisation de ces méthodes, bien que récente, n'est que rarement présente dans la littérature en relation avec la reconnaissance de radiographies dans un contexte COVID.</p> </p>", unsafe_allow_html=True)
    st.markdown("<p class='font'> Pour tenter de corriger ces biais, diverses méthodes de prétraitement ont été développées. Tout d'abord, l'application d'un masque pulmonaire a entraîné une réduction des performances du modèle mais une interprétabilité plus cohérente. L'application d'un masque pulmonaire a réduit de manière significative le biais d'image, mais les biais de luminosité et de forme étaient toujours présents. Pour réduire le biais de luminosité, différentes méthodes de modification de l'image ont été testées (correction gamma, stries de contraste, CLAHE). Au final, le meilleur résultat obtenu a été une précision de 0,89 après l'application d'un ajustement de la luminosité avec le modèle InceptionV3.</p> </p>", unsafe_allow_html=True)
    st.markdown("<p class='font'> Les prochains objectifs de ce projet seraient de tester d'autres combinaisons de modèles et de méthodes de prétraitement car de nombreuses méthodes présentes dans la littérature n'ont pas été testées. De plus, l'un des autres objectifs serait de tester cette approche sur d'autres jeux de données de rayons X afin d'évaluer la reproductibilité de l'approche et du modèle associé. </p>", unsafe_allow_html=True)
    image2 = Image.open(os.path.join(currentdir, 'data/article.png'))
    col1, col2,col3 = st.columns([1,1,1])
    with col1:
        st.write("")
    with col2:
        st.image(image2,width=800)
    with col3:
        st.write("")

    

