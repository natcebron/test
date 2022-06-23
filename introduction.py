import streamlit as st 
import os
import inspect                 #+Deployment

def app():

    st.markdown(f"""
                <style>
                  .reportview-container .main .block-container{{
                    padding-top: 0 rem;
                    marging-top: 0 rem;
                  }}
                  .reportview-container .main{{
                    padding-top: 0 rem;
                    marging-top: 0 rem;
                  }} 
                  .font {{
                    font-size:16px ; font-family: 'Arial'; color: #FFFFFF;}} 
               
                </style>
               """, unsafe_allow_html=True)
  
    st.markdown('''
          <h1>DEVELOPPEMENT D'UN MODELE DE DEEP LEARNING POUR LA CLASSIFICATION DE RADIOGRAPHIES DANS UN CONTEXTE COVID</h1>
          
          <p class="font">
            Le COVID-19 est une maladie infectieuse causée par le virus du SRAS-Cov-2. Ce virus affecte l'homme et a un tropisme pour les poumons. 
          </p>
          <p class="font">
          Le scanner et la radiographie du thorax, qui sont des outils d'imagerie de routine pour le diagnostic de la pneumonie, ont également été utilisés pour la détection des cas de COVID. Ils sont rapides et relativement faciles à réaliser en complément de l'examen clinique, sans être un test de détection virale à proprement parler. L'utilisation de ces technologies d'imagerie a conduit au développement de méthodes d'intelligence artificielle pour la détection automatique de virus à partir d'images pulmonaires.  
          </p>
          <p class="font">
          De nombreuses études sur la reconnaissance de rayons X basées sur l'apprentissage profond ont été réalisées. Dans une revue publiée en 2021 (Serena Low et al, 2021), 52 études publiées basées sur l'étude de radiographies ou de tomodensitogrammes entre 2019 et 2021 ont été résumées avec l'algorithme utilisé et les résultats obtenus. 
          </p>  
          <p class="font">
          Dans notre étude, nous avons utilisé le dataset COVID-QU-Ex présent sur Kaggle.          
          <p class="font">
           Lien jeu de données <a href="https://www.kaggle.com/datasets/anasmohammedtahir/covidqu">COVID-QU-Ex Dataset</a>.

          ''', unsafe_allow_html=True)
          

    

