import streamlit as st
import warnings
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import pandas as pd
import pywt
import scipy.signal
import sklearn.decomposition
import itertools
from sklearn.metrics import confusion_matrix
import sys, os, os.path
from fpdf import FPDF
import base64



def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css("style.css")

st.write("""
# **MyelinH Parkinson Analysis Interface**  
This system performs early diagnosis of parkinson's disease and monitoring of the patient health status.

""")

from PIL import Image
image = Image.open('logo-high-res.png')

st.sidebar.image(image, use_column_width=True)

st.sidebar.subheader('https://myelinh.com/')

task=st.selectbox('Select Task', ("Early Diagnosis", "Therapeutic Drug Monitoring", "General Health Status Monitoring"))

st.write(task)
if task=="Early Diagnosis":
    

    # Collects user input features into dataframe
    uploaded_file = st.file_uploader("Upload Brain Data for Disease Detection", type=["csv"])

    if uploaded_file is not None:
        patient0 = np.loadtxt(uploaded_file, delimiter=',')
        patient0 = BodyMirror.signal.butter_highpass(patient0, cutoff=0.1)
        patient0 = BodyMirror.signal.butter_bandpass(patient0, lo=lowcut, hi=highcut)
        patient0 = BodyMirror.signal.notch(patient0, cutoff=f0, Q=Q)
        patient0=patient0[:, 50*60:550*60]
        scaler2=BodyMirror.split.load_model('scaler_training_on_off.sav')
        patient0=scaler2.transform(patient0)
        spatial_filter=load_model('scaler_training_on_off.sav')
        features= extract_features(patient0, spatial_filter)
        classifier=load_model('modelKNN.sav')
        predict=classifier.clf.predict(features)
        prob= classifier.clf.predict_proba(features)
        if (predict==0):
            st.write('**Health Status**:', '**Detected Parkinson Patient**')
            
            #x='Detected Parkinson patient'
        else:
            st.write('** Health Status**:', '**Healthy Subject**')
            
            #x='Healthy subject'
elif task=="Therapeutic Drug Monitoring":
    

    #st.sidebar.header('Health Status')
    # Collects user input features into dataframe
    uploaded_file1 = st.file_uploader("Upload EEG Data for Health Status", type=["csv"])

    if uploaded_file1 is not None:
        patient1 = np.loadtxt(uploaded_file1, delimiter=',')
        patients1=BodyMirror.signal.butter_highpass(patient1, cutoff=0.1) 
        patient1 = BodyMirror.signal.butter_bandpass(patient1, lo=lowcut, hi=highcut)
        patient1 = BodyMirror.signal.notch(patient1, cutoff=f0, Q=Q)
        scaler=BodyMirror.split.load_model('scaler_training_on_off.sav')
        scaler2=BodyMirror.split.load_model('scaler_training_on_off.sav')
        patient1=scaler.transform(patient1)
        patient1=scaler2.transform(patient1)
        spatial_filter1=BodyMirror.split.load_model( 'scaler_training_on_off.sav')
        features1= extract_features1(patient1, spatial_filter1)
        classifier=BodyMirror.split.load_model('modelKNN.sav')
        predict1=classifier.clf.predict(features1)

        #predict1=neigh.predict(features1)
        if (predict1==1):
            st.write('**Patient Health Status**:', '**ON-Medication**')
            #x1='Patient"s health status: Off-medication'
        else:
            st.write('**Patient Health Status**:', '**OFF-Medication**')
            #x1='Patient"s health status: On-medication
        #st.write('**Current Health Status**:', x1)
        #st.write('**Probability**', prob1*100)

        
elif task=="General Health Status Monitoring":
    task2=st.selectbox('Select Task', ("Monitor Treatment Effectivness", "Monitor Current Status"))
    #b1=st.button('Monitor Treatment Effectivness', key="1")
    #b2=st.button('Monitor Patient Health Status', key="2")
    if task2=="Monitor Treatment Effectivness":
        
        uploaded_file2 = st.file_uploader("Upload Brain Data for S1 Health Status", type=["csv"])
        uploaded_file3 = st.file_uploader("Upload EEG Data for S2 Health Status", type=["csv"])
        if uploaded_file2 and uploaded_file3 is not None:

            data1 = np.loadtxt(uploaded_file2, delimiter=',')
            data2 = np.loadtxt(uploaded_file3, delimiter=',')
            st.subheader('Brain Frequency Bands Activity for S1')
            a1, a2, a3, a4, a5=plot_EEG_mean_frequency(data1, fs=250)
            st.subheader('Brain Frequency Bands Activity for S2')
            b1, b2, b3, b4, b5=plot_EEG_mean_frequency(data2, fs=250)
            st.subheader('Brain Frequency Bands Activity Comparison')
            #4703.688601711806, 15794.467536387318, 5718.347078219787, 13928.626680268217, 5434.85405394247
            #9043.59702333679,5811.136845088533, 12667.686540277266, 10371.506777649505, 11273.92599776178
            fig=plot_EEG_mean_frequency_comparison(a1, a2, a3, 
                                               a4, a5, 
                                               b1, b2, b3, 
                                               b4, b5, 'S1', 'S2')
            if a2> b2:
                output='Intepretation: Drug is effective and has reduced beta cortical oscillations'
                st.write('**Intepretation: Drug is effective and has reduced beta cortical oscillations**')
            else:
                st.write('**Intepretation: Drug is ineffective and has not reduced beta oscillations**')
                output='Intepretation: Drug is ineffective and has not reduced beta oscillations'
                
            
    elif task2=="Monitor Current Status":

            uploaded_file2 = st.file_uploader("Upload Brain Data for S1 Health Status", type=["csv"])
            uploaded_file3 = st.file_uploader("Upload EEG Data for S2 Health Status", type=["csv"])
            if uploaded_file2 and uploaded_file3 is not None:
                    

                    data1 = np.loadtxt(uploaded_file2, delimiter=',')
                    data2 = np.loadtxt(uploaded_file3, delimiter=',')
                    st.subheader('Brain Frequency Bands Activity for S1')
                    a1, a2, a3, a4, a5=plot_EEG_mean_frequency(data1, fs=250)
                    st.subheader('Brain Frequency Bands Activity for S2')
                    b1, b2, b3, b4, b5=plot_EEG_mean_frequency(data2, fs=250)
                    st.subheader('Brain Frequency Bands Activity Comparison')
                    #4703.688601711806, 15794.467536387318, 5718.347078219787, 13928.626680268217, 5434.85405394247
                    #9043.59702333679,5811.136845088533, 12667.686540277266, 10371.506777649505, 11273.92599776178
                    fig=plot_EEG_mean_frequency_comparison(a1/4703.688601711806, a2/15794.467536387318, a3/5718.347078219787, 
                                                       a4/13928.626680268217, a5/5434.85405394247, 
                                                       b1/9043.59702333679, b2/5811.136845088533, b3/12667.686540277266, 
                                                       b4/10371.506777649505, b5/11273.92599776178, 'S1', 'S2')
                    if a2/13928.626680268217 < b2/10371.506777649505:
                        output='Intepretation: Abnormal beta cortical oscillation caused by the disease'
                        st.write('**Intepretation: Abnormal beta cortical oscillation caused by the disease "Parkinson**')
                    else:
                        output='Intepretation: No Abnormal activity was detected'
                        


export_as_pdf = st.button("Export Medical Report")

def create_download_link(val, filename):
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="{filename}.pdf">Download file</a>'


if export_as_pdf:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(40, 10, output)
    fig.savefig('graph1.png', bbox_inches='tight')
    pdf.image('graph1.png', 32, 30, 160, 150)


    html = create_download_link(pdf.output(dest="S").encode("latin-1"), "test")

    st.markdown(html, unsafe_allow_html=True)

            