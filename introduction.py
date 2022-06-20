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
          <h2>ABOUT COVID</h2>
          <h3><b><i> What is COVID-19? </i></b></h3>   
          <p class="font">
            COVID-19 is an infectious disease caused by the SARS-Cov-2 virus. This virus affects humans and has a tropism for the lungs. 
          </p>
          <p class="font">
            This virus appeared at the end of 2019 and was the cause of a pandemic that upset the majority of countries and led to periods of confinement as was the case in France.  
          </p>
          <p class="font">
            This infection can be benign with very common symptoms (fever, cough or respiratory discomfort) but it can also cause more severe forms with cases of acute respiratory distress that can lead to the death of the patient. The rate of asymptomatic forms is estimated to be around 20% of infected persons. Transmission is by airborne spread of the virus.
Transmission increases in poorly ventilated indoor environments and when the infected person coughs, sneezes, talks or sings. The incubation period averages 5-6 days, with extremes ranging from two to fourteen days. Mortality occurs mainly in the elderly, with the average age of death from Covid-19 being 81 years. 
          </p>
          <p class="font">
           To fight against this virus, numerous vaccination campaigns have been carried out. From 2021, vaccines based on messenger RNA technology have been developed. This vaccination makes it possible to protect the population from serious forms of the disease but does not stop the spread of the virus. 
          </p>
          <h3><b><i> How to detect SARS-CoV-2 virus? </i></b></h3>   
          <p class="font">
          A diagnostic test for SARS-CoV-2 can be performed if there is a suspicion of coronavirus 2019 (Covid-19) disease on clinical examination. It can be performed by reverse transcriptase polymerase chain reaction tests for the detection of viral RNA (RT-PCR) or by ELISA antibody-based tests for the detection of virion proteins. 
          </p>
          <p class="font">
          Antigenic test is another method to detect this virus. These lateral flow tests are based on the detection of molecules (antigens) of the virus. These tests appeared in the year 2020 with quite satisfactory results and relieving the biological laboratories.
          </p>
          <p class="font">
          Chest CT and chest X-ray, which are routine imaging tools for the diagnosis of pneumonia, have also been used for the detection of COVID cases. They are quick and relatively easy to perform as an adjunct to the clinical examination, without being a viral detection test per se. The use of these imaging technologies has led to the development of artificial intelligence methods for automatic virus detection from lung images.  
          </p>
          <h3><b><i> Artificial intelligence for detection ? </i></b></h3> 
          <p class="font">
          Many studies on X-ray recognition based on deep learning have been carried out. In a review published in 2021 (Serena Low and al, 2021), 52 published studies based on the study of X-rays or CT scans between 2019 and 2021 were summarised with the algorithm used and the results obtained. 
          </p>  
          <h3><b><i> Project </i></b></h3> 
          <p class="font">
          In our study, we carried out an in-depth study of the images, which allowed us to determine the presence of bias between the different sets of images. These biases have a strong influence on the performance of the model. A confirmation of these biases was performed using dedicated methods such as grad cam or generative adversarial networks (GAN) but also via the use of voluntarily biased data sets. Image pre-processing methods have been implemented to try to correct or at least mitigate these biases. The results obtained are satisfactory (accuracy: 0.88) but still require improvement. The aim of this work was to set up a study methodology to detect and correct biases within datasets that could have a negative impact on the behaviour of a deep learning model.           </p> 
          <p class="font">
           Dataset link <a href="https://www.kaggle.com/datasets/anasmohammedtahir/covidqu">COVID-QU-Ex Dataset</a>.

          ''', unsafe_allow_html=True)
          

    

