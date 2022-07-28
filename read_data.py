
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline
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
from mne.datasets import sample
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    st.header("Data Selection")

    data_selection = st.radio("", ("Please Select Data Type", "EEG Brain Data", "EMG Muscle Data", "ECG Heart Data", "fMRI Data"))
    if data_selection=="EEG Brain Data":
        st.header("Brain Data Analysis")
        task2=st.selectbox('Please Select Data Format', ( "", "MyelinH Device", "BrainVision- (.vhdr) ", "European data format- (.edf)", "BioSemi data format-(.bdf)", "General data format (.gdf)", "Neuroscan CNT (.cnt)", "EGI simple binary (.egi)", "EEGLAB files (.set, .fdt)", "EGI MFF (.mff)", "Nicolet (.data)", "eXimia EEG data (.nxe)", "Persyst EEG data (.lay, .dat)", "Nihon Kohden EEG data (.eeg, .21e, .pnt, .log)", "XDF data (.xdf, .xdfz)" ))
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
                    #epochs_data=mne.read_epochs_eeglab(uploaded)
                    #epochs_data.set_eeg_reference('average', projection=True)
                    #epochs_data.apply_proj()
                    #montage=raw_data.get_montage()
                    #epochs_data.set_montage(montage)
                    #info=epochs_data.info
                    epochs_h=mne.read_epochs("epochs_h.fif")
                    epochs_p=mne.read_epochs("epochs_p.fif") 
                        
                    
                    st.header("Digital Biomarkers Extraction")
                    tab11, tab22, tab33, tab444= st.tabs([ "Patient", "Control Subject", "Comparison", "Classification"])   
                    with tab11:
                        st.header("Patient")

                        tab1, tab2, tab3, tab5 = st.tabs([ "Time-Analysis", "Frequency-Analysis", "Time-Frequency", "Source-Localization"])     
                        with tab1:
                            import Time_Analysis
                            Time_Analysis.time_analysis_p(epochs_p)
                        with tab2:
                            import Frequency_Analysis
                            Frequency_Analysis.frequency_analysis(epochs_h)
                        with tab3:
                            import Time_Frequency_Analysis
                            Time_Frequency_Analysis.time_frequency(epochs_p)
 
                        with tab5:

                            import Source_Localization
                            
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
                            #epochs_data1=mne.read_epochs_eeglab(uploaded1)
                            #epochs_data.set_eeg_reference('average', projection=True)
                            #epochs_data.apply_proj()
                            #montage=raw_data.get_montage()
                            #epochs_data.set_montage(montage)
                            #info1=epochs_data1.info
                            epochs_h=mne.read_epochs("epochs_h.fif")
                            tab14, tab24, tab34, tab54 = st.tabs([ "Time-Analysis", "Frequency-Analysis", "Time-Frequency", "Source-Localization"])     
                            with tab14:
                                import Time_Analysis
                                Time_Analysis.time_analysis_h(epochs_h)
                            with tab24:
                                import Frequency_Analysis
                                Frequency_Analysis.frequency_analysis_h(epochs_h)
                            with tab34:
                                import Time_Frequency_Analysis
                                Time_Frequency_Analysis.time_frequency_h(epochs_h)
               
                            with tab54:

                                import Source_Localization
                        else:
                            epochs_h=mne.read_epochs("epochs_h.fif")
                            epochs_p=mne.read_epochs("epochs_p.fif") 
                            #epochs_data1=epochs_data
                            #info1=info 

                    with tab33:
                        if uploaded_file and uploaded_file1 is not None: 
                            def compare(epochs_p, epochs_h):
                                    evoked_p = epochs_p.average()
                                    evoked_p.apply_proj()
                                    evoked_h= epochs_h.average()
                                    evoked_h.apply_proj()

                                    evokeds_list= [evoked_h, evoked_p]
                                    conds = ('Healthy', 'Patient')
                                    evks = dict(zip(conds, evokeds_list))

                                    def custom_func(x):
                                        return x.max(axis=1)
                                    #dict(aud=evoked_h, vi=evoked_p)
                                    mne.viz.plot_compare_evokeds(evks, colors=dict(Healthy=0, Patient=1),
                                           linestyles=dict(Healthy='solid', Patient='dashed'))

                                    for combine in ('mean', 'gfp', custom_func):
                                        figures=mne.viz.plot_compare_evokeds(evks, picks='eeg', combine=combine)
                                    fig4=figures[0]
                                    #fig5=figures[1]
                                    st.pyplot(fig4)
                                    #st.pyplot(fig5)
                                    fig6=mne.viz.plot_compare_evokeds(evks, picks='eeg', colors=dict(Healthy=0, Patient=1),
                                           linestyles=dict(Healthy='solid', Patient='dashed'),
                                     axes='topo', styles=dict(Healthy=dict(linewidth=1),
                                                              Patient=dict(linewidth=1)))
                                    st.pyplot(fig6[0])
                                    
                            compare(epochs_p, epochs_h)
                       
                        else:
                            st.error("Please upload the data first") 
                            
                    with tab444:
                        if uploaded_file and uploaded_file1 is not None: 
                            def classify(epochs_p, epochs_h):
                                X1 = epochs_h.get_data()[:,:, :]  # MEG signals: n_epochs, n_meg_channels, n_times
                                y1 = epochs_h.events[:, 2]  # target: auditory left vs visual left

                                X2 = epochs_p.get_data()[:,:, :]   # MEG signals: n_epochs, n_meg_channels, n_times
                                y2 = np.zeros(X2.shape[0])  # target: auditory left vs visual left
                                X=np.concatenate((X1, X2), axis=0)
                                y=np.concatenate((y1, y2), axis=0) 
                                scores = []
                                cv = ShuffleSplit(10, test_size=0.2, random_state=42)
                                cv_split = cv.split(X)

                                # Assemble a classifier
                                lda = LinearDiscriminantAnalysis()
                                csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

                                # Use scikit-learn Pipeline with cross_val_score function
                                clf = Pipeline([('CSP', csp), ('LDA', lda)])
                                scores = cross_val_score(clf, X**2, y, cv=cv, n_jobs=None)

                                # Printing the results
                                class_balance = np.mean(y == y[0])
                                class_balance = max(class_balance, 1. - class_balance)
                                #print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores),
                                                                                          #class_balance))

                                # plot CSP patterns estimated on full data for visualization
                                f=csp.fit_transform(X**2, y)

                                st.pyplot(csp.plot_patterns(epochs_h.info, ch_type='eeg', units='Patterns (AU)', size=1.5))
                                if scores != []:
                                        for _ in stqdm(range(5)):
                                                sleep(0.5)
                                        st.success('Data has been successfully classified')
                                        results=np.mean(scores)*100
                                        resu=str(results)
                                        resu="Classification Accuracy:" + resu+  "%"
                                        st.header(resu)
                                    
                                    

                                    
                            classify(epochs_p, epochs_h)
                       
                        else:
                            st.error("Please upload the data first") 
                    
                        
                else:
                    st.warning('Data has not been uploaded yet')







   
