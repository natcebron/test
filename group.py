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
import glob
import numpy as np
from matplotlib import pyplot as plt

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  
    local_css(os.path.join(currentdir, "style.css"))
    #Préparation de la page
    st.markdown(""" <style> .font {font-size:16px ; font-family: 'Arial'; color: #FFFFFF;} </style> """, unsafe_allow_html=True)
    st.markdown("# COMPARAISON DES GROUPES EN FONCTION DES CARACTÉRISTIQUES DE L'IMAGE")
    st.markdown('<p class="font"> Before the development of a deep learning model, a comparison of the different sets of images was carried out. To do this, a study of the characteristics of the images was performed). </p>', unsafe_allow_html=True)
    sns.set_theme(style="white", palette=None)

####################
# DISTRIBUTION DES PIXELS 
####################
    st.markdown('## DISTRIBUTION DES PIXELS')

# retrieving data from file.

    loaded_arr_normal = np.loadtxt(os.path.join(currentdir, 'data/arr_reshaped_Normal.txt'))
    loaded_arr_COVID = np.loadtxt(os.path.join(currentdir, 'data/arr_reshaped_COVID.txt'))
    loaded_arr_NC = np.loadtxt(os.path.join(currentdir, 'data/arr_reshaped_NC.txt'))

# This loadedArr is a 2D array, therefore
# we need to convert it to the original
# array shape.reshaping to get original
# matrice with original shape.
    load_original_arr_normal = loaded_arr_normal.reshape(loaded_arr_normal.shape[0], loaded_arr_normal.shape[1] // 3, 3)
    load_original_arr_COVID = loaded_arr_COVID.reshape(loaded_arr_COVID.shape[0], loaded_arr_COVID.shape[1] // 3, 3)
    load_original_arr_NC = loaded_arr_NC.reshape(loaded_arr_NC.shape[0], loaded_arr_NC.shape[1] // 3, 3)

    st.markdown("<p class='font'> La première étape a consisté à comparer la distribution des pixels entre les différents ensembles d'images. Cette étude de distribution a été réalisée sur une image moyenne créée à partir de toutes les images d'un ensemble. </p>", unsafe_allow_html=True)

    fig = plt.figure(figsize=(20, 7))
    plt.rcParams.update({'text.color': "white",'axes.labelcolor': "white",'text.color':'white','xtick.color':'white','ytick.color':'white','axes.facecolor':'#0e1117','axes.edgecolor':'#0e1117'})
    fig.set_facecolor("#0e1117")
    arr_COVID = np.array(load_original_arr_normal).ravel()
    arr_NC = np.array(load_original_arr_COVID).ravel()
    arr_Normal = np.array(load_original_arr_NC).ravel()

    sns.kdeplot(arr_COVID,color='red',label='COVID')# Can show all four figures at once by calling plt.show() here, outside the loop.
    sns.kdeplot(arr_NC,color='green',label='NC')# Can show all four figures at once by calling plt.show() here, outside the loop.
    sns.kdeplot(arr_Normal,color='blue',label='Normal')# Can show all four figures at once by calling plt.show() here, outside the loop.
    plt.title("Distribution de l'intensité du pixel",fontsize = 20);
    plt.xlabel('Intensité de pixels');
    plt.ylabel('Densité');
    plt.legend();
    plt.show()

    st.pyplot(fig)
    st.markdown("<p class='font'> La distribution de l'intensité des pixels est très différente entre les différents groupes. En effet, les images du groupe Non-COVID montrent une distribution symétrique avec un pic à 140 alors que le groupe Normal a un pic à 120. Pour le groupe COVID, la distribution ne présente pas de pic. Cette différence de distribution peut avoir un impact et sera donc prise en compte lors des étapes de prétraitement des images.", unsafe_allow_html=True)

#############################################
# PIXEL METRICS : MEAN AND STANDARD DEVIATION
#############################################

    st.markdown('## Métrique des pixels (moyenne et écart-type)')
    st.markdown("<p class='font'>Pour aller plus loin, une étude plus détaillée des pixels a été réalisée, incluant une détermination des métriques de la moyenne des pixels et de l'écart-type des pixels. Ces valeurs ont été déterminées sur chaque image de chaque ensemble.  </p>", unsafe_allow_html=True)

    df = pd.read_csv(os.path.join(currentdir, 'data/Mean and std analysis.csv'))

    fig2 = plt.figure(figsize=(20, 7))
    plt.rcParams.update({'text.color': "white",'axes.labelcolor': "white",'text.color':'white','xtick.color':'white','ytick.color':'white','axes.facecolor':'#0e1117','axes.edgecolor':'#0e1117'})
    fig2.set_facecolor("#0e1117")
    df_COVID=df[df['group'] == "COVID"]
    df_NCOVID=df[df['group'] == "Non-COVID"]
    df_Normal=df[df['group'] == "Normal"]


    sns.kdeplot(df_COVID['mean'],color='red',label='COVID')# Can show all four figures at once by calling plt.show() here, outside the loop.
    sns.kdeplot(df_NCOVID['mean'],color='green',label='Non-COVID')# Can show all four figures at once by calling plt.show() here, outside the loop.
    sns.kdeplot(df_Normal['mean'],color='blue',label='Normal')# Can show all four figures at once by calling plt.show() here, outside the loop.
    plt.axvline(x = 115, color = 'g', linestyle = '-') 
    plt.axvline(x = 120, color = 'b', linestyle = '-') 
    plt.axvline(x = 145, color = 'r', linestyle = '-') 
    plt.text(105,0.0215,'115',color='g', fontsize=14)
    plt.text(122,0.0215,'120',color='b', fontsize=14)
    plt.text(150,0.0215,'145',color='r', fontsize=14)

    plt.title("Moyenne",fontsize = 20);
    plt.xlabel('Intensité de pixel');
    plt.ylabel('Densité');
    plt.legend();
    plt.show()

    st.pyplot(fig2)
    st.markdown("<p class='font'> Pour tous les ensembles, la distribution des valeurs moyennes des pixels est symétrique avec la présence d'un pic. Ce pic a une valeur plus élevée pour le groupe COVID avec une valeur de 140 par rapport aux groupes Normal et Non-COVID avec des valeurs de 120 et 115 respectivement. L'hypothèse que nous pouvons émettre est que les images du groupe COVID ont une luminosité, un contraste ou une saturation différents par rapport aux deux autres groupes, mais cela reste à confirmer. </p>", unsafe_allow_html=True)

    fig3 = plt.figure(figsize=(20, 7))
    plt.rcParams.update({'text.color': "white",'axes.labelcolor': "white",'text.color':'white','xtick.color':'white','ytick.color':'white','axes.facecolor':'#0e1117','axes.edgecolor':'#0e1117'})
    fig3.set_facecolor("#0e1117")

    
#create your own color array
    my_colors = ["green", "red", "blue"]
  
# add color array to set_palette
# function of seaborn
    sns.set_palette( my_colors )
  
    sns.boxplot(x=df['std'],y=df['group'], data=df)
    plt.title("Ecart-type",fontsize = 20)
    st.pyplot(fig3)
    st.markdown("<p class='font'> L'étude de la variable pixel de l'écart-type a été réalisée. Cette métrique a été représentée sous forme de boxplot. La valeur de l'écart-type semble être plus faible dans le groupe COVID avec une valeur moyenne de 55 par rapport au groupe Normal qui a une valeur de 63. Le groupe Non-COVID se situe entre ces deux ensembles avec une valeur de 58. Les images du groupe COVID auraient donc une variabilité des pixels plus faible.   </p>", unsafe_allow_html=True)


    st.markdown('## Forme des poumons')
    image = Image.open(os.path.join(currentdir, 'data/forme.png'))
   
    col1, mid, col2 = st.columns([0.5,2,0.5])
    with mid:
        st.image(image, width=512)
