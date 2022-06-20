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
          <h2>Conclusion</h2>
          <p class="font">
          In this study, the Gradcam deep learning model interpretability method,  was used to understand how the model classifies different image sets. This method confirmed the hypothesis that the dataset studied was biased. Indeed, we showed that the model on the basic images classified the images using non-targeted parts on the lungs. This result was confirmed using other methods (biased dataset and generative adversarial networks). The use of these methods, although recent, are only rarely found in the literature in relation to the recognition of radiographs in a COVID context.
          </p>
          <p class="font">
          In an attempt to correct for these biases, various pre-processing methods have been developed. Firstly, the application of a lung mask resulted in reduced model performance but more consistent interpretability. The application of a lung mask significantly reduced image bias, but brightness and shape bias were still present. To reduce the brightness bias, different image modification methods were tested (gamma correction, contrast streaking, CLAHE). In the end, the best result obtained was 0.88 accuracy after applying contrast streching with InceptionV3 model.
          </p>
          <p class="font">
          The next objectives of this project would be to test other combinations of model and pre-processing methods as many methods present in the literature have not been tested. Furthermore, one of the other objectives would be to test this approach on other X-rays datasets to evaluate the reproducibility of the approach and the associated model.

          </p>
          ''', unsafe_allow_html=True)
          

    

