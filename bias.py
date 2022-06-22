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
    st.markdown("# DETERMINATION DES BIAIS")

    st.markdown('## PREMIER RESULTAT')
    st.markdown("<p class='font'> Cette section est consacrée à la détermination du biais. Dans un premier temps, un modèle basé sur l'apprentissage par transfert a été testé sur les radiographies brutes. Nous obtenons les résultats suivants :  </p>", unsafe_allow_html=True)

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

    st.markdown("<p class='font'> Les résultats obtenus dans ce premier modèle sont très satisfaisants, notamment pour le groupe COVID, avec des valeurs de précision de 0,99. Pour vérifier que le modèle n'est pas biaisé, nous avons réalisé une étude Grad-CAM sur les images vraies positives et fausses négatives pour chaque groupe.  </p>", unsafe_allow_html=True)

    st.markdown('## GRAD-CAM')
    image = Image.open(os.path.join(currentdir, 'data/Gradcam.png'))
    new_image = image.resize((1400, 1000))

    st.markdown("<p class='font'> Grad-Cam est une méthode publiée en 2016 qui vise à savoir quelles parties de l'image ont été utilisées par le modèle pour classer les images. Le résultat est affiché sous la forme d'une carte thermique.  </p>", unsafe_allow_html=True)

    st.image(image,width=1000)
    st.markdown("<p class='font'> En utilisant cette méthode nous obtenons des résultats très concluants que notre modèle est biaisé. En effet, nous pouvons voir que les zones les plus utilisées par le modèle (en jaune) ne correspondent pas aux poumons mais à des parties externes de l'image. Ce résultat est encore plus évident pour le groupe COVID où l'on constate que ce sont toujours les mêmes zones qui sont trouvées. Ce résultat explique notre précision pour ce groupe très élevé. </p>", unsafe_allow_html=True)


    st.markdown('## DATASETS BIAISES')

    st.markdown('## GANS')


