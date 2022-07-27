
import os

import numpy as np

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from datetime import datetime
import streamlit as st
import time

import json

from time import sleep
import streamlit as st
import codecs, json
from stqdm import stqdm


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from datetime import datetime
import streamlit as st

import json

from time import sleep
import streamlit as st
import codecs, json
from stqdm import stqdm

import sys, os, os.path

import numpy as np

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score
import mne
from mne import make_forward_solution
from mne.minimum_norm import make_inverse_operator, apply_inverse
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import mne
from mne.datasets import sample
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)

from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
# features, so the resulting filters used are spatio-temporal
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import mne
from mne.datasets import sample
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
#%matplotlib qt
from pylab import rcParams

#import tensorflow as tf
#from tensorflow.python.lib.io import file_io
#import myelinh_functions
#import login_app
#import visualization
from PIL import Image
import base64
import mpld3
from mpld3 import plugins
#import matplotlib.pyplot as mpld3
import streamlit.components.v1 as components

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
def data_read():

    data_selection = st.radio("", ("Please Select Data Type", "EEG Brain Data", "EMG Muscle Data", "ECG Heart Data", "fMRI Data"))
    if data_selection=="EEG Brain Data":
        st.header("Brain Data Analysis")
        task2=st.selectbox('Please Select Data Format', ( "", "BrainVision- (.vhdr) ", "European data format- (.edf)", "BioSemi data format-(.bdf)", "General data format (.gdf)", "Neuroscan CNT (.cnt)", "EGI simple binary (.egi)", "EEGLAB files (.set, .fdt)", "EGI MFF (.mff)", "Nicolet (.data)", "eXimia EEG data (.nxe)", "Persyst EEG data (.lay, .dat)", "Nihon Kohden EEG data (.eeg, .21e, .pnt, .log)", "XDF data (.xdf, .xdfz)", "Mat file" ))
        if task2=="EEGLAB files (.set, .fdt)":

            #file_name = st.text_input('Please enter the path where your data is stored')
                #st.warning('Please make sure that your data is stored in the same location as Myelin-H platform')
                uploaded_file = st.file_uploader("Choose a file of your patient")

                if uploaded_file is not None:
                    for _ in stqdm(range(5)):
                            sleep(0.5)
                    st.success('Data has been uploaded succesfully')
                    uploaded=uploaded_file.name
                    raw_data=mne.read_epochs_eeglab(uploaded)
                    #raw_data=raw_data.set_eeg_reference('average', projection=True)
                    epochs_data=mne.read_epochs_eeglab(uploaded)
                    #epochs_data.set_eeg_reference('average', projection=True)
                    #epochs_data.apply_proj()
                    #montage=raw_data.get_montage()
                    #epochs_data.set_montage(montage)
                    info=epochs_data.info
                        
                    
                    st.header("Data Analysis for Digital Biomarkers")
                    tab11, tab22= st.tabs([ "Patient", "Control Subject"])   
                    with tab11:
                        st.header("Patient")

                        tab1, tab2, tab3, tab4, tab5 = st.tabs([ "Time-Analysis", "Frequency-Analysis", "Time-Frequency", "Pattern-Recognition", "Source-Localization"])     
                        with tab1:
                            import Time_Analysis
                            Time_Analysis.time_analysis_p(epochs_data)
                        with tab2:
                          st.write("done")
                            #import Frequency_Analysis
                        with tab3:
                          st.write("done")
                            #import Time_Frequency_Analysis
                        with tab4:
                          st.write("done")
                            #import Pattern_Recognition
                        with tab5:
                           st.write("done")

                            #import Source_Localization  
                    with tab22:
                        st.header("Control Subject")
                        uploaded_file1 = st.file_uploader("Choose a file of your control subject")
                        if uploaded_file1 is not None:
                            for _ in stqdm(range(5)):
                                    sleep(0.5)
                            st.success('Data has been uploaded succesfully')
                            uploaded1=uploaded_file1.name
                            raw_data1=mne.read_epochs_eeglab(uploaded1)
                            #raw_data=raw_data.set_eeg_reference('average', projection=True)
                            epochs_data1=mne.read_epochs_eeglab(uploaded1)
                            #epochs_data.set_eeg_reference('average', projection=True)
                            #epochs_data.apply_proj()
                            #montage=raw_data.get_montage()
                            #epochs_data.set_montage(montage)
                            info1=epochs_data1.info
                            tab14, tab24, tab34, tab44, tab54 = st.tabs([ "Time-Analysis", "Frequency-Analysis", "Time-Frequency", "Pattern-Recognition", "Source-Localization"])     
                            with tab14:
                                import Time_Analysis
                                Time_Analysis.time_analysis_h(epochs_data1)
                            with tab24:
                              st.write("done")
                                #import Frequency_Analysis
                            with tab34:
                              st.write("done")
                                #import Time_Frequency_Analysis
                            with tab44:
                              st.write("done")
                                #import Pattern_Recognition
                            with tab54:
                              st.write("done")

                                #import Source_Localization
                        else:
                            epochs_data1=epochs_data
                            info1=info 

                else:
                    st.warning('Data has not been uploaded yet')





   
