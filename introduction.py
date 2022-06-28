
import streamlit as st
import streamlit.components.v1 as components
import os                      #+Deployment
import inspect                 #+Deployment
#importing all the necessary libraries
import pandas as pd
import numpy as np                     
import matplotlib.pyplot as plt
import os
import random               #+Deployment

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  
    local_css(os.path.join(currentdir, "style.css"))
    #Préparation de la page
    st.markdown(""" <style> .font {font-size:16px ; font-family: 'Arial'; color: #FFFFFF;} </style> """, unsafe_allow_html=True)
    st.markdown("# DEVELOPPEMENT D'UN MODELE DE DEEP LEARNING POUR LA CLASSIFICATION DE RADIOGRAPHIES DANS UN CONTEXTE COVID")
    st.markdown("<p class='font'> Le COVID-19 est une maladie infectieuse causée par le virus du SRAS-Cov-2. Ce virus affecte l'homme et a un tropisme pour les poumons. </p>", unsafe_allow_html=True)
    st.markdown("<p class='font'> Le scanner et la radiographie du thorax, qui sont des outils d'imagerie de routine pour le diagnostic de la pneumonie, ont également été utilisés pour la détection des cas de COVID. Ils sont rapides et relativement faciles à réaliser en complément de l'examen clinique, sans être un test de détection virale à proprement parler. L'utilisation de ces technologies d'imagerie a conduit au développement de méthodes d'intelligence artificielle pour la détection automatique de virus à partir d'images pulmonaires.  </p>", unsafe_allow_html=True)
    st.markdown("<p class='font'> De nombreuses études sur la reconnaissance de rayons X basées sur l'apprentissage profond ont été réalisées. Dans une revue publiée en 2021 (Serena Low et al, 2021), 52 études publiées basées sur l'étude de radiographies ou de tomodensitogrammes entre 2019 et 2021 ont été résumées avec l'algorithme utilisé et les résultats obtenus. </p>", unsafe_allow_html=True)
    st.markdown("<p class='font'> Dans notre étude, nous avons utilisé le dataset COVID-QU-Ex présent sur Kaggle.          </p>", unsafe_allow_html=True)

 
          

    

