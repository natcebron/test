import streamlit as st
import streamlit.components.v1 as components
import os                      #+Deployment
import inspect                 #+Deployment
#importing all the necessary libraries
import pandas as pd
import numpy as np                     
import matplotlib.pyplot as plt
import os
import cv2 as cv
import random
import seaborn as sns
from PIL import Image, ImageStat
import matplotlib.image as mpimg
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def app():
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  
    local_css(os.path.join(currentdir, "style.css"))
    #Préparation de la page
    st.markdown(""" <style> .font {font-size:20px ; font-family: 'Arial'; color: #FFFFFF;} </style> """, unsafe_allow_html=True)
    st.markdown("# GROUP COMPARISON BY PICTURE CHARACTERISTICS")
    st.markdown('<p class="font"> Before the development of a deep learning model, a comparison of the different sets of images was carried out. To do this, a study of the characteristics of the images was performed). </p>', unsafe_allow_html=True)

####################
# PIXEL DISTRIBUTION 
####################
    st.markdown('## Pixel distribution')

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

    st.markdown('<p class="font"> The first step was to compare the distribution of pixels between the different image sets. This distribution study was carried out on an "average image" created from all the images in a set. </p>', unsafe_allow_html=True)

    fig = plt.figure(figsize=(20, 7))
    plt.rcParams.update({'text.color': "white",'axes.labelcolor': "white",'text.color':'white','xtick.color':'white','ytick.color':'white','axes.facecolor':'#0e1117','axes.edgecolor':'#0e1117'})
    fig.set_facecolor("#0e1117")
    arr_COVID = np.array(load_original_arr_normal).ravel()
    arr_NC = np.array(load_original_arr_COVID).ravel()
    arr_Normal = np.array(load_original_arr_NC).ravel()

    sns.kdeplot(arr_COVID,color='red',label='COVID')# Can show all four figures at once by calling plt.show() here, outside the loop.
    sns.kdeplot(arr_NC,color='green',label='NC')# Can show all four figures at once by calling plt.show() here, outside the loop.
    sns.kdeplot(arr_Normal,color='blue',label='Normal')# Can show all four figures at once by calling plt.show() here, outside the loop.
    plt.title("Pixel intensity distribution",fontsize = 20);
    plt.xlabel('Pixel intensity');
    plt.ylabel('Density');
    plt.legend();
    plt.show()

    st.pyplot(fig)
    st.markdown('<p class="font"> The pixel intensity distribution is very different between the different sets. Indeed, the images of the Non-COVID group show a symmetrical distribution with a peak at 140 while the Normal group has a peak at 120. For the COVID group, the distribution does not have a peak. This difference in distribution may have an impact and will therefore be taken into account during the image pre-processing stages.', unsafe_allow_html=True)

#############################################
# PIXEL METRICS : MEAN AND STANDARD DEVIATION
#############################################

    st.markdown('## Pixel metrics (mean and std)')
    st.markdown('<p class="font">To go further, a more detailed study of the pixels was carried out, including a determination of the pixel mean and pixel standard deviation metrics. These values were determined on each image of each set.  </p>', unsafe_allow_html=True)

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

    plt.title("Mean",fontsize = 20);
    plt.xlabel('Pixel intensity');
    plt.ylabel('Density');
    plt.legend();
    plt.show()

    st.pyplot(fig2)
    st.markdown('<p class="font"> For all sets, the distribution of pixel mean values are symmetrical with the presence of a peak. This peak has a higher value for the COVID group with a value of 140 compared to the Normal and Non-COVID groups with values of 120 and 115 respectively. The hypothesis we can make is that the images in the COVID set have a different brightness, contrast or saturation compared to the other two sets but this remains to be confirmed. </p>', unsafe_allow_html=True)

    fig3 = plt.figure(figsize=(20, 7))
    plt.rcParams.update({'text.color': "white",'axes.labelcolor': "white",'text.color':'white','xtick.color':'white','ytick.color':'white','axes.facecolor':'#0e1117','axes.edgecolor':'#0e1117'})
    fig3.set_facecolor("#0e1117")

    
#create your own color array
    my_colors = ["green", "red", "blue"]
  
# add color array to set_palette
# function of seaborn
    sns.set_palette( my_colors )
  
    sns.boxplot(x=df['std'],y=df['group'], data=df)
    plt.title("Standard deviation",fontsize = 20)
    st.pyplot(fig3)
    st.markdown('<p class="font"> The study of the standard deviation pixel variable was carried out. This metric was represented as a boxplot. The value of standard deviation seems to be lower in the COVID group with a mean value of 55 compared to the Normal group which has a value of 63. The Non-COVID group falls between these two sets with a value of 58. The images in the COVID set would therefore have a lower pixel variability.  </p>', unsafe_allow_html=True)

##################
# BRIGHTNESS STUDY
##################

    st.markdown('## Brightness study')
    st.markdown('<p class="font">The study of the luminosity between the different sets was carried out. This metric was measured on each of the images and a representation in the form of a density curve was made.  </p>', unsafe_allow_html=True)

    df5 = pd.read_csv(os.path.join(currentdir, 'data/Brightness.csv'))
    fig4 = plt.figure(figsize=(20, 7))
    plt.rcParams.update({'text.color': "white",'axes.labelcolor': "white",'text.color':'white','xtick.color':'white','ytick.color':'white','axes.facecolor':'#0e1117','axes.edgecolor':'#0e1117'})
    fig4.set_facecolor("#0e1117")
    sns.kdeplot(df5['COVID'],color='red',label='COVID');# Can show all four figures at once by calling plt.show() here, outside the loop.
    sns.kdeplot(df5['NC'],color='green',label='NC');# Can show all four figures at once by calling plt.show() here, outside the loop.
    sns.kdeplot(df5['Normal'],color='blue',label='Normal');# Can show all four figures at once by calling plt.show() here, outside the loop.
    plt.title("Brightness",fontsize = 20)
    plt.xlabel('Brightness value');

    plt.legend();

    st.pyplot(fig4)
    st.markdown('<p class="font">The density curves of the 3 sets are similar with a larger peak for the Non-COVID group. In addition, some images from the normal group show very high brightness values (200-250) compared to the peaks.  </p>', unsafe_allow_html=True)


################
# CONTRAST STUDY
################
    st.markdown('## Contrast study')
    st.markdown('<p class="font"> The contrast study was also carried out. This was determined for each of the images by comparing the maximum and minimum pixel values for each of the images and then represented as a density curve.</p>', unsafe_allow_html=True)

    df6 = pd.read_csv(os.path.join(currentdir, 'data/Contrast.csv'))
    fig5 = plt.figure(figsize=(20, 7))
    plt.rcParams.update({'text.color': "white",'axes.labelcolor': "white",'text.color':'white','xtick.color':'white','ytick.color':'white','axes.facecolor':'#0e1117','axes.edgecolor':'#0e1117'})
    fig5.set_facecolor("#0e1117")
    df6_COVID=df6[df6['group'] == "COVID"]
    df6_NCOVID=df6[df6['group'] == "Non-COVID"]
    df6_Normal=df6[df6['group'] == "Normal"]


    sns.kdeplot(df6_COVID['contrast'],color='red',label='COVID')# Can show all four figures at once by calling plt.show() here, outside the loop.
    sns.kdeplot(df6_NCOVID['contrast'],color='green',label='Non-COVID')# Can show all four figures at once by calling plt.show() here, outside the loop.
    sns.kdeplot(df6_Normal['contrast'],color='blue',label='Normal')# Can show all four figures at once by calling plt.show() here, outside the loop.

    plt.title("Contrast",fontsize = 20);
    plt.xlabel('Contrast value');
    plt.ylabel('Density');
    plt.legend();
    plt.show()
    st.pyplot(fig5)
    st.markdown('<p class="font">Les courbes de densité des 3 ensembles sont similaires avec un pic plus important pour le groupe Normal. </p>', unsafe_allow_html=True)

########################
# OUTLIERS DETERMINATION
########################
    st.markdown('## Outliers determination')
    st.markdown('<p class="font"> An outlier study was carried out. For this purpose the interquartile range (IQR) method was used on the previously determined standard deviation values. The results are presented in the form of a scatter plot according to the standard deviation and the mean of the pixels. If a value is considered to be an outlier then the size of the point on the scatter plot will be larger. </p>', unsafe_allow_html=True)

    df6 = pd.read_csv(os.path.join(currentdir, 'data/outliers.csv'))
    Q1 = df6['O_stddev'].quantile(0.25)
    Q3 = df6['O_stddev'].quantile(0.75)
    IQR = Q3 - Q1
    df6['outliers'] = (df6['O_stddev'] < (Q1 - 1.5 * IQR)) |(df6['O_stddev'] > (Q3 + 1.5 * IQR))

    fig6 = plt.figure(figsize=(20, 7))
    plt.rcParams.update({'text.color': "white",'axes.labelcolor': "white",'text.color':'white','xtick.color':'white','ytick.color':'white','axes.facecolor':'#0e1117','axes.edgecolor':'#0e1117'})
    fig6.set_facecolor("#0e1117")
    df6['outliers'] = df6['outliers'].replace((True, False), (10, 1))

    ax = sns.scatterplot(data=df6, x="O_mean", y='O_stddev', hue = 'group',size='outliers',alpha=0.8,palette=['green','red','blue']);
    plt.title('Representation of the images as a function of the standard deviation and the mean',fontsize = 20);
    plt.xlabel('Mean');
    plt.ylabel('Standard deviation');

    st.pyplot(fig6)
    st.markdown('<p class="font">The COVID group has a large number of outliers (340) compared to the Normal and Non-COVID sets which have 4 and 12 outliers respectively. The presence of these outliers may have an impact on the performance of the model and should be taken into account.  </p>', unsafe_allow_html=True)

    st.markdown('## Forme des poumons ??')

    return None
