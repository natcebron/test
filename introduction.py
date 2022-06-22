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
                    font-size:20px ; font-family: 'Arial'; color: #FFFFFF;}} 
               
                </style>
               """, unsafe_allow_html=True)
  
    st.markdown('''
          <h1>DEVELOPPEMENT D'UN MODELE DE DEEP LEARNING POUR LA CLASSIFICATION DE RADIOGRAPHIES DANS UN CONTEXTE COVID</h1>
          <h2>MEMBRES : Damien BRUZZONE, Nathan CEBRON, Matthieu PELINGRE</h2>
          
          <h4><b><i> Qu'est-ce que le COVID-19 ? </i></b></h4>   
          <h4><b><i> Comment détecter le virus SARS-CoV-2 ? </i></b></h4>   

          <p class="font">
            Le COVID-19 est une maladie infectieuse causée par le virus du SRAS-Cov-2. Ce virus affecte l'homme et a un tropisme pour les poumons. 
          </p>
          <p class="font">
            Apparu fin 2019, ce virus est à l'origine d'une pandémie qui a bouleversé la majorité des pays et entraîné des périodes d'enfermement comme ce fut le cas en France.  
          </p>
          <p class="font">
            Cette infection peut être bénigne avec des symptômes très courants (fièvre, toux ou gêne respiratoire) mais elle peut aussi provoquer des formes plus sévères avec des cas de détresse respiratoire aiguë pouvant conduire au décès du patient. Le taux de formes asymptomatiques est estimé à environ 20% des personnes infectées. La transmission se fait par voie aérienne du virus.
            La transmission augmente dans les environnements intérieurs mal ventilés et lorsque la personne infectée tousse, éternue, parle ou chante. La période d'incubation est en moyenne de 5 à 6 jours, avec des extrêmes allant de deux à quatorze jours. La mortalité survient principalement chez les personnes âgées, l'âge moyen de décès par Covid-19 étant de 81 ans. 
          </p>
          <p class="font">
           Pour lutter contre ce virus, de nombreuses campagnes de vaccination ont été menées. A partir de 2021, des vaccins basés sur la technologie de l'ARN messager ont été développés. Cette vaccination permet de protéger la population des formes graves de la maladie mais n'arrête pas la propagation du virus. 
          </p>
          <h4><b><i> Comment détecter le virus SARS-CoV-2 ? </i></b></h4>   
          <p class="font">
          Un test de diagnostic du SRAS-CoV-2 peut être effectué en cas de suspicion de maladie à coronavirus 2019 (Covid-19) à l'examen clinique. Il peut être réalisé par des tests d'amplification en chaîne par polymérase avec transcriptase inverse pour la détection de l'ARN viral (RT-PCR) ou par des tests ELISA à base d'anticorps pour la détection des protéines des virions. 
          </p>
          <p class="font">
          Le test antigénique est une autre méthode pour détecter ce virus. Ces tests à flux latéral sont basés sur la détection de molécules (antigènes) du virus. Ces tests sont apparus en 2020 avec des résultats tout à fait satisfaisants et soulageant les laboratoires de biologie.
          </p>
          <p class="font">
          Le scanner et la radiographie du thorax, qui sont des outils d'imagerie de routine pour le diagnostic de la pneumonie, ont également été utilisés pour la détection des cas de COVID. Ils sont rapides et relativement faciles à réaliser en complément de l'examen clinique, sans être un test de détection virale à proprement parler. L'utilisation de ces technologies d'imagerie a conduit au développement de méthodes d'intelligence artificielle pour la détection automatique de virus à partir d'images pulmonaires.  
          </p>
          <h3><b><i> L'intelligence artificielle pour la détection ? </i></b></h3> 
          <p class="font">
          De nombreuses études sur la reconnaissance de rayons X basées sur l'apprentissage profond ont été réalisées. Dans une revue publiée en 2021 (Serena Low et al, 2021), 52 études publiées basées sur l'étude de radiographies ou de tomodensitogrammes entre 2019 et 2021 ont été résumées avec l'algorithme utilisé et les résultats obtenus. 
          </p>  
          <h4><b><i> Projet </i></b></h4> 
          <p class="font">
          Dans notre étude, nous avons réalisé une étude approfondie des images, ce qui nous a permis de déterminer la présence de biais entre les différents ensembles d'images. Ces biais ont une forte influence sur la performance du modèle. Une confirmation de ces biais a été effectuée à l'aide de méthodes dédiées telles que le grad cam ou les réseaux antagonistes  génératifs (GAN) mais aussi via l'utilisation de jeux de données volontairement biaisés. Des méthodes de prétraitement d'images ont été mises en œuvre pour tenter de corriger ou au moins d'atténuer ces biais. Les résultats obtenus sont satisfaisants (précision : 0,88) mais nécessitent encore des améliorations. L'objectif de ce travail était de mettre en place une méthodologie d'étude pour détecter et corriger les biais au sein de jeux de données qui pourraient avoir un impact négatif sur le comportement d'un modèle d'apprentissage profond.          
          <p class="font">
           Lien jeu de données <a href="https://www.kaggle.com/datasets/anasmohammedtahir/covidqu">COVID-QU-Ex Dataset</a>.

          ''', unsafe_allow_html=True)
          

    

