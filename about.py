import streamlit as st 
import os
import base64
import streamlit.components.v1 as components
import os                      #+Deployment
import inspect                 #+Deployment

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url, size=50):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" height={size}px/>
        </a>'''
    return html_code


def app():

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 


    c1, c2, c3  = st.columns([0.5, 1, 1])
    with c1:
        st.markdown(f'''<u>Project members</u> :''', unsafe_allow_html=True)  
    with c2:
        logo_linkedin = get_img_with_href(os.path.join(currentdir, 'data/linkedin.png'), 'https://www.linkedin.com/in/damien-bruzzone-7b2836148/', 20)
        st.markdown(f'''<a href="https://www.linkedin.com/in/damien-bruzzone-7b2836148/" style="text-decoration: none;color:white">Damien BRUZZONE</a> {logo_linkedin}''', unsafe_allow_html=True) 
    with c3:
        logo_linkedin = get_img_with_href(os.path.join(currentdir, 'data/linkedin.png'), 'https://www.linkedin.com/in/nathan-cebron-9992a1a4/', 20)
        st.markdown(f'''<a href="https://www.linkedin.com/in/nathan-cebron-9992a1a4/" style="text-decoration: none;color:white">Nathan CEBRON</a> {logo_linkedin}''', unsafe_allow_html=True)       
  
    c1, c2, c3  = st.columns([0.5, 1, 1])
    with c2:
        logo_linkedin = get_img_with_href(os.path.join(currentdir, 'data/linkedin.png'), 'https://www.linkedin.com/in/matthieu-pelingre-3667b0210/', 20)
        st.markdown(f'''<a href="https://www.linkedin.com/in/matthieu-pelingre-3667b0210/" style="text-decoration: none;color:white">Jos&eacute; Matthieu PELINGRE</a> {logo_linkedin}''', unsafe_allow_html=True) 


    c1, c2 = st.columns([0.5, 2])
    c1.markdown(f'''<u>Project mentor</u> :''', unsafe_allow_html=True)  
    logo_linkedin = get_img_with_href(os.path.join(currentdir, 'data/linkedin.png'), 'https://www.linkedin.com/in/gaspard-grimm/', 20)
    c2.markdown(f'''<a href="https://www.linkedin.com/in/gaspard-grimm/" style="text-decoration: none;color:white">Gaspard GRIMM (DataScientest)</a> {logo_linkedin}''', unsafe_allow_html=True)    

    c1, c2 = st.columns([0.5, 2])
    c1.markdown(f'''<u>Github</u> :''', unsafe_allow_html=True)  
    c2.markdown(f'''<a href="https://github.com/DataScientest-Studio/DeMACIA-RX">DeMACIA-RX project</a>''', unsafe_allow_html=True) 

