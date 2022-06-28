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
        st.dataframe(data=df)

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
    st.markdown('### Impact de la luminosité')
    st.markdown("<p class='font'> Comme nous avons pu le voir dans la partie de visualisation des données, certains paramètres comme la luminosité et la forme des poumons varient selon les classes et pourraient donc potentiellement induire des biais lors de l’entraînement des modèles. </p>", unsafe_allow_html=True)

    img = Image.open(os.path.join(currentdir, 'data/graph1.png'))

    st.image(img,width=1000)
    st.markdown("<p class='font'> Afin de détecter ces biais, nous nous sommes inspirés des travaux de Schaaf et al. (DOI :10.48550/arXiv.210700360)  et avons volontairement introduit des biais dans deux ensembles de tests, en modifiant la luminosité des images masquées, tout en restant dans le domaine d’études. Pour l’ensemble Normal-Up, la luminosité de la classe Normal uniquement a été augmentée de 20%. Pour l’ensemble COVID-Down, la luminosité de la classe COVID-19 uniquement a été diminuée de 10%  </p>", unsafe_allow_html=True)
    img = Image.open(os.path.join(currentdir,"data/graph2.PNG"))

    st.image(img, width=1000)
    st.markdown("<p class='font'> Les différents ensembles ont été testés sur un modèle LeNet5, entraîné sur le dataset original et présentant un biais lié à la luminosité des images. Comme les deux nouveaux ensembles sont similaires à l’ensemble de test original (mêmes images) hormis les différences de luminosités, il est aisé d’évaluer facilement les changements induits par les ensembles biaisés. </p>", unsafe_allow_html=True)
    img = Image.open(os.path.join(currentdir,"data/table1.png"))

    col1, col2, col3 = st.columns([0.7,4,0.5])

    with col1:
        st.write(' ')

    with col2:
        st.image(img, width=700,caption='Table 1. Comparaison des résultats obtenus pour les ensembles biaisés sur LeNet5')

    with col3:
        st.write(' ')

    st.markdown("<p class='font'> Ainsi, une augmentation de la luminosité sur la classe Normal amènera le modèle à affecter un plus grand nombre d’images à la classe COVID-19, mettant en lumière le biais de luminosité lié à ces deux classes. Dans le cas d’une diminution de la luminosité de la classe COVID-19, le changement est réparti entre les classes   Non-COVID   et   Normal.   Cette   différence   pourrait   être   due   à   une discrimination selon la forme des poumons. </p>", unsafe_allow_html=True)
    st.markdown("<p class='font'> Nous avons ensuite testé différents modèles et paramètres pour tenter de réduire l’impact de ce biais : </p>", unsafe_allow_html=True)
    st.markdown("   - **Encodage** : l’utilisation d’images en RGB **diminue** drastiquement l’impact du biais")
    st.markdown("- **Dimensions   de   l’image** :  augmenter   les   dimensions   des   images **diminue** l’impact du biais")
    st.markdown("- **Prétraitement** : l’utilisation de la fonction de prétraitement spécifique au modèle  **diminue**  l’impact du biais par rapport à un simple rescale par 1/255")
    st.markdown("- **Fine-tuning** :  l’effet dépend du modèle et des classes (ResNet50 et DenseNet201 : **diminue** l’impact sur la classe Normal et l’augmente sur la classe COVID-19 ; légère augmentation globale pour InceptionV3)")
    img = Image.open(os.path.join(currentdir,"data/table2.png"))

    col1, col2, col3 = st.columns([0.7,4,0.5])

    with col1:
        st.write(' ')

    with col2:
        st.image(img, width=700,caption="Table 2. Modifications observées par l'utilisation d'ensembles biaisés sur un modèle Inceptionv3")

    with col3:
        st.write(' ')

    
    st.markdown("<p class='font'> Les modèles InceptionV3 semblent les plus robustes au biais de luminosité et les meilleurs résultats sont obtenu sur ce modèle, entraîné par fine-tuning, avec une accuracy globale de 92% :</p>", unsafe_allow_html=True)
    st.markdown("<p class='font'> Nous pouvons constater que les changements de luminosité affectent assez peu le modèle (modifications inférieures à 2.3% la plupart du temps). Nous avons pu constater, sur les modèles utilisés en transfert learning, que la luminosité n’était pas la principale source de biais. La forme des poumons a en effet un impact beaucoup plus significatif</p>", unsafe_allow_html=True)
    st.markdown('### Impact de la forme des poumons')

    st.markdown("<p class='font'>Comme nous avons pu le voir précédemment, des divergences liées à la forme des poumons entre les différentes classes pourraient induire des biais lors de l’entraînement de nos modèles. Nous avons d’ailleurs pu le constater sur un modèle DenseNet201 (accuracy globale : 92%) entraîné par fine-tuning, en appliquant des zooms et dézooms sur l’ensemble de test : </p>", unsafe_allow_html=True)
    
    img = Image.open(os.path.join(currentdir,"data/table3.png"))
    col1, col2, col3 = st.columns([0.7,4,0.5])

    with col1:
        st.write(' ')

    with col2:
        st.image(img, width=700,caption="Table 3. Modifications observées par l'utilisation d'ensembles zoomés et dézoomés sur un modèle Inceptionv3")

    with col3:
        st.write(' ')
    


    st.markdown("<p class='font'> Comme nous avons pu le voir précédemment, des divergences liées à la forme des poumons entre les différentes classes pourraient induire des biais lors de l’entraînement de nos modèles. Nous avons d’ailleurs pu le constater sur un modèle DenseNet201 (accuracy globale : 92%) entraîné par fine-tuning, en appliquant des zooms et dézooms sur l’ensemble de test :</p>", unsafe_allow_html=True)
    st.markdown("<p class='font'> En regardant plus en détail, il est possible de constater que la classe Normal est la plus affectée dans les deux cas. Cela pourrait être lié à la faible déviation entre les différentes formes des poumons de cette classe dans le dataset. Le moindre changement de forme entraînera donc une forte modification des prédictions pour cette classe. Pour mesurer l’impact de la forme des poumons, nous avons donc, tout d’abord, tenté d’entraîner un modèle pour qu’il ne reconnaisse que le forme des poumons. Un modèle InceptionV3 a donc été entraîné en n’utilisant que les masques fournis avec le dataset. La luminosité de ces derniers a été diminuée aléatoirement entre 20 et 70% pour représenter le domaine d’étude :</p>", unsafe_allow_html=True)
    img = Image.open(os.path.join(currentdir,"data/graph3.PNG"))

    st.image(img, width=1000)
    
    
    st.markdown("<p class='font'> Le modèle entraîné obtient des résultats décents avec une précision globale de 75 %, et les f1-scores suivants : 72 % pour la classe COVID-19 et 77 % pour les classes   Non-COVID   et   Normal.   Nous   avons   ensuite   testé   ce   modèle   sur l’ensemble   de   test   original   et   nous   n’avons   pu   constater   que   de   faibles changements avec une accuracy de 75 sur cet ensemble, et des f1-scores de 71% pour la classe COVID-19, 75% pour la classe Non-COVID et 74% pour la classe Normal. Il est donc tout à fait possible d’entraîner un modèle à ne reconnaître que les formes des poumons. La présence d’un biais lié à la forme est donc possible et peut avoir un poids non négligeable. Notons les résultats observés sur les ensembles biaisés Normal-Up et COVID-Down sont très similaires à ceux obtenus avec l’ensemble de test original sur ce modèle (0.6% de différence au maximum). Le biais lié à la luminosité pour ce modèle est donc très faible.</p>", unsafe_allow_html=True)
    st.markdown("<p class='font'> Nous avons donc essayé d’utiliser l’ensemble de test des masques ainsi créé pour mesurer l’impact de la forme sur les prédictions du modèle InceptionV3 obtenu dans la partie sur l’impact de la luminosité. Sur cet ensemble, le modèle atteint une précision globale de 43% (contre 93%) et les scores f1 suivants : 44% pour COVID-19, 49% pour Non-COVID et 26% pour la classe normale. Comme prévu, les résultats obtenus sont proches de ceux qui pourraient être obtenus dans   une   prédiction   aléatoire   (33%),   mais   soulignent   qu'une   partie   des prédictions pourrait encore être liée à la forme du poumon. Ceci est encore plus évident lorsque l'on observe la matrice de confusion. </p>", unsafe_allow_html=True)
    img = Image.open(os.path.join(currentdir,"data/table4.png"))
    col1, col2, col3 = st.columns([0.7,4,0.5])

    with col1:
        st.write(' ')

    with col2:
        st.image(img, width=700,caption="Table 4. Matrice de confusion d'Inceptionv3 sur l'ensemble de test des masques")

    with col3:
        st.write(' ')
    
    st.markdown("<p class='font'>Nous pouvons observer que la distribution n'est pas aussi aléatoire que nous aurions pu le penser. En effet, la classe Normal est sous représentée dans les prédictions. Cela peut être dû à un biais lié à la faible déviation des formes des poumons entre les images de cette classe. La majorité des prédictions pour cette classe sont reportées sur les classes COVID-19  et Non-COVID. Pour  ces  dernières,  la forme  seule  semble avoir beaucoup plus d'impact. Pour la classe Non-COVID, en particulier, le nombre de bonnes prédictions (1390) est largement supérieur aux faux négatifs de cette classe, attribués à la classe COVID-19 (688). Par conséquent, l'hypothèse d'un biais lié à la forme du poumon pour ce modèle ne peut être rejetée. Des résultats similaires peuvent être observés pour les modèles DenseNet201 et ResNet50, avec parfois une absence totale de prédiction pour la classe Normal. </p>", unsafe_allow_html=True)
    st.markdown('### Conclusion')
    st.markdown("<p class='font'> Nous avons pu voir qu’il était possible d’introduire des biais dans des ensembles de test pour mesurer l’impact de ces derniers sur les prédictions de nos modèles. Il devient alors possible de modifier les paramètres d’entraînement dans le but de minimiser les différences entre ces ensembles de test et l’original, comme pour Normal-Up et COVID-Down, ou d’obtenir des prédictions les plus aléatoires possibles, comme avec les masques.  Nous avons également pu constater que, pour les modèles étudiés dans cette partie, le biais de luminosité est moindre comparé au biais lié à la forme des poumons. Ce dernier devra donc être réglé en priorité, tout en continuant à surveiller l’impact du biais de luminosité pour éviter que celui-ci ne devienne, lui aussi, problématique.</p>", unsafe_allow_html=True)

    st.markdown('## GANS')


