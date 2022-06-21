import streamlit as st 

import os 
os.chdir("C:/Users/natha/DeMACIA-RX-main/Streamlit")


import introduction                     #+Deployment
import group
import bias
import exploration
import model
import home
import about
import inspect
import prediction
import conclusion
import about
import __init__
import test2
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
logo = os.path.join(currentdir, 'ressources/covid_21.png')
PAGE_CONFIG = {"page_title":"DeMACIA-RX.io","page_icon": logo,"layout":"wide"}
st.set_page_config(**PAGE_CONFIG)

MENU = {
    "Introduction" : introduction,
    "Dataset exploration" : exploration,
    "Group comparison" : group,
    "Bias determination" : bias,
    "Model" : model,
    "Prediction" : prediction,
    "Conclusion" : conclusion,
    "About":about,
    "test":__init__, 
    "test2":test2       
}

st.sidebar.title('Menu')
selection_page = st.sidebar.radio("",list(MENU.keys()))
page = MENU[selection_page]
page.app()