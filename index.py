import streamlit as st 

import os 


import introduction                     #+Deployment
import group
import bias_presentation2
import exploration
import model
import home
import about
import inspect
import prediction
import conclusion2
import about
import __init__
import test2
import config
import member
from collections import OrderedDict

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
logo = os.path.join(currentdir, 'data/covid_1.png')
PAGE_CONFIG = {"page_title":"DeMACIA-RX.io","page_icon": logo,"layout":"wide"}
st.set_page_config(**PAGE_CONFIG)

SOMMAIRE = {
    "Introduction" : introduction,
    "Dataset exploration" : exploration,
    "Group comparison" : group,
    "Bias determination" : bias_presentation2,
    "Model" : model,
    "Prediction" : prediction,
    "Prediction interactive v1":test2,
    "Prediction interactive v2":__init__,
     "Conclusion" : conclusion2

    
}




st.sidebar.title('SOMMAIRE')
selection_page = st.sidebar.radio("",list(SOMMAIRE.keys()))
page = SOMMAIRE[selection_page]
page.app()

st.sidebar.markdown("### Team members:")
for member in config.TEAM_MEMBERS:
        st.sidebar.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

st.sidebar.markdown("### Mentor : Gaspard GRIMM")
        
st.sidebar.markdown("### Promotion : DS AVRIL 2022")
