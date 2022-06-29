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


    st.markdown("<p class='font'> Dans cette étude, différentes méthodes ont été testées afin d'interpréter les résultats du modèle, de détecter les biais et de mesurer leur impact. </p>", unsafe_allow_html=True)
    with st.expander("Biais détectés et méthodes"):
        st.markdown("- biais lié au backgroud par GradCAM")
        st.markdown("- biais de luminosité par les sets de test biaisés")
        st.markdown("- biais de forme par GradCAM et sets de test biaisé")
        st.markdown("- biais lié à un dataset déséquilibré (COVID-QU)")
    st.markdown("<p class='font'> L'utilisation de ces méthodes, bien que récente, n'est que rarement présente dans la littérature, notament pour la détection du COVID en Deep Learning. </p>", unsafe_allow_html=True)
    st.markdown("")  # espace

    st.markdown("<p class='font'> Pour tenter de corriger ces biais, diverses méthodes de prétraitement ont été étudiées. </p>", unsafe_allow_html=True)
    with st.expander("Méthodes de prétraitement par biais"):
        st.markdown("- biais lié au backgroud : segmentation des poumons (modèle UNet)")
        st.markdown("- biais de luminosité : correction gamma, stries de contraste, CLAHE ...")
        st.markdown("- biais de forme : rotations aléatoires, zooms, dézooms ...")
        st.markdown("- biais lié à un dataset déséquilibré : génération d'images via réseau GAN")
    st.markdown("<p class='font'> Le meilleur résultat obtenu a été une précision de 0,89 après segmentation et l'application d'un ajustement de la luminosité avec le modèle InceptionV3. </p>", unsafe_allow_html=True)
    st.markdown("")  # espace

    st.markdown("<p class='font'> Les prochains objectifs de ce projet seraient de tester d'autres combinaisons de modèles et de méthodes de prétraitement. </p>", unsafe_allow_html=True)
    with st.expander("Perspectives"):
        st.markdown("- tester d'autres méthodes de prétraitement présentes dans la littérature")
        st.markdown("- tester cette approche sur d'autres jeux de données (reproductibilité méthode + polyvalence modèle)")
        st.markdown("- GAN : augmenter résolution des images (plus de resources et temps d'entraînement plus long)")

    image2 = Image.open(os.path.join(currentdir, 'data/article.png'))
    col1, col2,col3 = st.columns([0.5,4,0.5])
    with col1:
        st.write("")
    with col2:
        st.image(image2,width=800)
    with col3:
        st.write("")

    

