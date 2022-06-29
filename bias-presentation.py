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
    st.markdown(""" <style> .font {font-size:16px ; font-family: 'Arial'; color: #FFFFFF;} </style> """, unsafe_allow_html=True)
    st.markdown("# DETERMINATION DES BIAIS")

    st.markdown('## PREMIER RESULTAT')
    st.markdown("<p class='font'> Cette section est consacrée à la détermination du biais. Dans un premier temps, un modèle basé sur l'apprentissage par transfert a été testé sur les radiographies brutes. Nous obtenons les résultats suivants :  </p>", unsafe_allow_html=True)

    # initialize list of lists
    data = [['COVID', 0.99,0.90,0.94], ['Normal',0.89,0.92,0.90], ['Non_COVID',0.87,0.92,0.89],['Average',0.91,0.91,0.91]]
    st.markdown("<p class='font'> Paramètres modèles : InceptionV3, epochs = 30, loss_function = sparse_categorical_crossentropy, optimizer = Adam </p>", unsafe_allow_html=True)

 
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns=['Group', 'Precision','Recall','F1-score'])
    


    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:
        st.dataframe(data=df,width=800)

    with col3:
        st.write(' ')

    st.markdown("<p class='font'> Les résultats obtenus dans ce premier modèle sont très satisfaisants, notamment pour le groupe COVID, avec des valeurs de précision de 0,99. Pour vérifier que le modèle n'est pas biaisé, nous avons réalisé une étude Grad-CAM sur les images vraies positives et fausses négatives pour chaque groupe.  </p>", unsafe_allow_html=True)

    st.markdown('## GRAD-CAM')
    image = Image.open(os.path.join(currentdir, 'data/Gradcam.png'))
    new_image = image.resize((1400, 1000))

    st.markdown("<p class='font'> Grad-Cam est une méthode publiée en 2016 qui vise à savoir quelles parties de l'image ont été utilisées par le modèle pour classer les images. Le résultat est affiché sous la forme d'une carte thermique.  </p>", unsafe_allow_html=True)

    st.markdown("<p class='font'> En utilisant cette méthode nous obtenons des résultats très concluants que notre modèle est biaisé. En effet, nous pouvons voir que les zones les plus utilisées par le modèle (en rouge) ne correspondent pas aux poumons mais à des parties externes de l'image. Ce résultat est encore plus évident pour le groupe COVID où l'on constate que ce sont toujours les mêmes zones qui sont trouvées. Ce résultat explique notre précision pour ce groupe très élevé. </p>", unsafe_allow_html=True)
    st.image(image,width=1000)


    st.markdown('## DATASETS BIAISES')
    st.markdown("<p class='font'> Plusieurs paramètres pourraient induire des biais lors de l’entraînement : </p>", unsafe_allow_html=True)
    st.markdown("- Luminosité des images")
    st.markdown("- Forme des poumons")
    st.markdown("<p class='font'> Inspiré de la littérature (Schaaf et al. DOI: 10.48550/arXiv.210700360) -> création de datasets comportant des biais. </p>", unsafe_allow_html=True)

    st.markdown('### I. Impact de la luminosité')
    st.markdown("<p class='font'> Distributions des luminosités différentes entres les classes, même avec les masques. </p>", unsafe_allow_html=True)

    img = Image.open(os.path.join(currentdir, 'data/graph1a.png'))
    img2 = Image.open(os.path.join(currentdir, 'data/graph1b.png'))

    col1, col2 = st.columns(2)

    with col1:
        st.image(img,caption='Distribution de la luminosité (Radiographies complètes)')


    with col2:
        st.image(img2,caption='Distribution de la luminosité (Radiographies masquées)')
    st.markdown("")  # espace

    st.markdown("##### I.1 Création de deux ensembles de tests biaisé :")
    st.markdown("- Normal-Up : luminosité + 20% sur classe Normal")
    st.markdown("- COVID-Down : luminosité - 10% sur classe COVID-19")
    st.markdown("- choix des % : rester dans le domaine d'étude")

    img = Image.open(os.path.join(currentdir,"data/graph2a.png"))
    img2 = Image.open(os.path.join(currentdir,"data/graph2b.png"))
    img3 = Image.open(os.path.join(currentdir,"data/graph2c.png"))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(img,caption='Distribution de pixels (original)')

    with col2:
        st.image(img2,caption='Distribution de pixels (Normal Up)')

    with col3:
        st.image(img3,caption='Distribution de pixels (COVID-Down)')
    st.markdown("")  # espace

    st.markdown("##### I.2 Evaluation sur un LeNet5 biaisé :")
    st.markdown("- Entrainement sur le dataset original")
    st.markdown("- Comparaison des changement induis sur les classes modifiés des ensembles biaisés")
    img = Image.open(os.path.join(currentdir,"data/table1.png"))

    col1, col2, col3 = st.columns([0.7,4,0.5])

    with col1:
        st.write("")

    with col2:
        st.image(img,caption='Table 1. Comparaison des résultats obtenus pour les ensembles biaisés sur LeNet5')

    with col3:
        st.write("")

    with st.expander("Observations"):
        st.markdown("- Normal-Up : augmenter la luminosité sur la classe Normal -> augmente prédictions COVID-19")
        st.markdown("- COVID-Down : diminuer la luminosité de la classe COVID-19 -> répartitions entre Non-COVID et Normal")

    with st.expander("Conclusions"):
        st.markdown("- mise en lumière du biais de luminosité du modèle")
        st.markdown("- les différences entres les nouvelles répartitions pourraient être liées à la forme des poumons")

    st.markdown("")  # espace

    st.markdown("##### I.3 Autres modèles et paramètres :")
    st.markdown("<p class='font'> Test de différents paramètres et modèles (LeNet5, DenseNet201, ResNet50 et InceptionV3) pour tenter de réduire l’impact de ce biais : </p>", unsafe_allow_html=True)
    st.markdown("- **Encodage** : l’utilisation d’images en RGB **diminue** drastiquement l’impact du biais")
    st.markdown("- **Résolution** : augmenter la résolution des images **diminue** l’impact du biais")
    st.markdown("- **Prétraitement** : l’utilisation de la fonction de prétraitement spécifique au modèle  **diminue**  l’impact du biais par rapport à un simple rescale par 1/255")
    st.markdown("- **Fine-tuning** : l’effet dépend du modèle et des classes (ResNet50 et DenseNet201 : **diminue** l’impact sur la classe Normal et l’augmente sur la classe COVID-19 ; légère augmentation globale pour InceptionV3)")
    st.markdown("<p class='font'> </p>", unsafe_allow_html=True)

    st.markdown("<p class='font'> InceptionV3 semble le plus robuste au biais de luminosité et les meilleurs résultats sont obtenu sur ce modèle, entraîné par fine-tuning :</p>", unsafe_allow_html=True)

    img = Image.open(os.path.join(currentdir,"data/table2.png"))

    col1, col2, col3 = st.columns([0.7,4,0.5])

    with col1:
        st.write(' ')

    with col2:
        st.image(img, width=700,caption="Table 2. Modifications observées par l'utilisation d'ensembles biaisés sur un modèle Inceptionv3")

    with col3:
        st.write(' ')

    
    st.markdown("<p class='font'> Les changements de luminosité affectent peu le modèle (< 2.3%). </p>", unsafe_allow_html=True)
    with st.expander("Généralité Transfert Learning"):
        st.markdown("- Luminosité : pas la principale source de biais")
        st.markdown("- Forme des poumons : impact beaucoup plus significatif")
    st.markdown("")  # espace

    st.markdown('### II. Impact de la forme des poumons')

    st.markdown("<p class='font'> Pour un DenseNet201 (acc = 92%), en appliquant des zooms et dézooms sur l'ensemble d'entraînement : </p>", unsafe_allow_html=True)
    
    img = Image.open(os.path.join(currentdir,"data/table3.png"))
    col1, col2, col3 = st.columns([0.7,4,0.5])

    with col1:
        st.write(' ')

    with col2:
        st.image(img, width=700,caption="Table 3. Modifications observées par l'utilisation d'ensembles zoomés et dézoomés sur un modèle Inceptionv3")

    with col3:
        st.write(' ')

    with st.expander("Observation : la classe Normal est la plus affectée dans les deux cas."):
        st.markdown("- lié à la faible déviation des formes de la classe")
        st.markdown("- changements -> modification des prédictions pour cette classe")
    st.markdown("")

    st.markdown("##### II.1 Modèle volontairement biaisé sur la forme : ")
    st.markdown("- InceptionV3 (fine-tuning)")
    st.markdown("- entraînement sur les masques uniquement")
    st.markdown("- diminution aléatoire de la luminosité des masques entre 20 et 70% (reste dans domaine d’étude)")
    img = Image.open(os.path.join(currentdir,"data/graph3a.png"))
    img2 = Image.open(os.path.join(currentdir,"data/graph3b.png"))

    col1, col2 = st.columns(2)

    with col1:
        st.image(img,caption='Distribution de la luminosité(Ensemble de test original)')

    with col2:
        st.image(img2,caption='Distribution de la luminosité(masques, luminosité aléatoire)')

    with st.expander("Scores du modèle sur l'ensemble de test des masques"):
        st.markdown("- accuracy globale : 75%")
        st.markdown("- f1-score COVID-19 : 72%")
        st.markdown("- f1-score Non-COVID : 77%")
        st.markdown("- f1-score Normal : 77%")

    with st.expander("Scores du modèle sur l'ensemble de test original"):
        st.markdown("- accuracy globale : 73%")
        st.markdown("- f1-score COVID-19 : 71%")
        st.markdown("- f1-score Non-COVID : 75%")
        st.markdown("- f1-score Normal : 74%")


    with st.expander("Conclusions"):
        st.markdown("- Il est possible d’entraîner un modèle à ne reconnaître que les formes des poumons.")
        st.markdown("- La présence d’un biais lié à la forme est possible et peut avoir un poids non négligeable.")
        st.markdown("- Pas d'impact de la luminosité sur le modèle (différences < 0.6% sur Normal-Up et COVID-Down)")
        st.markdown("NB: l'utilisation de paramètres zoom_range, rotation_range, shear_range, ... sur ce modèle permet de mesurer leur impact direct sur le biais")
    st.markdown("")

    st.markdown("##### II.2 L’ensemble de test des masques pour mesurer le biais de forme : ")

    st.markdown("<p class='font'> Evaluation du modèle InceptionV3 précédent sur l'ensemble de test des masques </p>", unsafe_allow_html=True)

    with st.expander("Scores InceptionV3 sur les masques"):
        st.markdown("- accuracy globale : 43% (contre 93%)")
        st.markdown("- f1-score COVID-19 : 44% (contre 93%)")
        st.markdown("- f1-score Non-COVID : 49% (contre 92%)")
        st.markdown("- f1-score Normal : 26% (contre 92%)")

    st.markdown("<p class='font'> Les résultats obtenus sont proches d'une prédiction aléatoire (33%), mais soulignent qu'une partie des prédictions pourrait encore être liée à la forme du poumon. </p>", unsafe_allow_html=True)
    st.markdown("<p class='font'> Ceci est encore plus évident lorsque l'on observe la matrice de confusion : </p>", unsafe_allow_html=True)



    img = Image.open(os.path.join(currentdir,"data/table4.png"))
    col1, col2, col3 = st.columns([0.7,4,0.5])

    with col1:
        st.write(' ')

    with col2:
        st.image(img, width=700,caption="Table 4. Matrice de confusion d'Inceptionv3 sur l'ensemble de test des masques")

    with col3:
        st.write(' ')

    with st.expander("Observations"):
        st.markdown("- la distribution n'est pas totalement aléatoire")
        st.markdown("- la classe Normal est sous représentée dans les prédictions (lié à la faible déviation des formes)")
        st.markdown("- les prédictions pour cette classe sont reportées sur les classes COVID-19 et Non-COVID")
        st.markdown("- la forme seule semble avoir beaucoup plus d'impact sur ces deux classes")
        st.markdown("- Non-COVID : vrai positifs (1390) > faux négatifs attribués à la classe COVID-19 (688) ")

    with st.expander("Conclusions"):
        st.markdown("- L'hypothèse d'un biais lié à la forme du poumon pour ce modèle ne peut être rejetée.")
        st.markdown("- Réduction du biais à l'aide de zoom_range, rotation_range ... (augmentation de l'aléatoire)")
        st.markdown("NB : Résultats similaires pour DenseNet201 et ResNet50 (parfois pas de prédiction classe Normal).")
    st.markdown("")

    st.markdown('### III. Conclusion')
    st.markdown("- il est possible d’introduire des biais dans des ensembles de test pour mesurer leur impact")
    st.markdown("- il est alors possible de modifier les paramètres d’entraînement dans le but de les minimiser")
    st.markdown("- l'impact des paramètres sur les biais peut être mesuré en les faisant varier un à un")
    st.markdown("- pour les modèles en TL étudiés ici, l'impact de la luminosité est moindre p/r à la forme")

    st.markdown("<p class='font'> Le biais de forme devra donc être réglé en priorité, tout en continuant à surveiller l’impact du biais de luminosité pour éviter que celui-ci ne devienne, à son tour, problématique.</p>", unsafe_allow_html=True)

    st.markdown('## GANS')


if __name__ == '__main__':  # test
    app()
