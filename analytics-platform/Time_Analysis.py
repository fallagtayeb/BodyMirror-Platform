import streamlit as st
#from streamlit.hashing import _CodeHasher
from streamlit.legacy_caching.hashing import _CodeHasher

# Get a reference to the auth service
import pyrebase
import json
from firebase import firebase
from getpass import getpass

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
sys.path.append('/BodyMirror')
import numpy as np
import BodyMirror
print('Welcome to BodyMirror version {}'.format(BodyMirror.version.__version__))
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from google.cloud import storage
import pickle
from sklearn.ensemble import RandomForestClassifier
import scipy.linalg as la
from io import BytesIO

import sys, os, os.path
import pyvista as pv
from pyvistaqt import BackgroundPlotter
sys.path.append('/BodyMirror')
import numpy as np
import BodyMirror
print ('Welcome to BodyMirror version {}'.format (BodyMirror.version.__version__))
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
rcParams['figure.figsize'] = 12, 8
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

#import myelinh_functions
#import text_to_speech

def time_analysis(epochs_p):

    epochs_h=mne.read_epochs("epochs_h.fif")

    evoked_h = epochs_h.average()
    evoked_h.apply_proj()

    evoked_p = epochs_p.average()
    evoked_p.apply_proj()


    task = st.selectbox('Select Task', ['', "Visualize Collected Data", "Topographic Maps", "Analyze Specific Electrodes", "Temporal Analysis"])

    times1=[0, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 
            0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.2, 0.21, .22, 0.25, 0.28, 0.35]



    #state = _get_state()


    #set_png_as_page_bg('2.jpg')
    if task == "Visualize Collected Data":
        tab1, tab2 = st.tabs([ "Time Analysis", "Frequency Analysis", "Time Frequency", "Pattern Recognition", "Source Localization"]) 
        st.title("Healthy Subjects")
        #fig1, anim_c3= evoked.animate_topomap('eeg', times=times1, blit=False, frame_rate=1)
        fig1, anim_c3= evoked_h.animate_topomap('eeg', times=times1, blit=False, frame_rate=1)
        #st.video(anim_c3)
        anim_c3.save(r'Animation1.mp4')
        #HtmlFile = line_ani.to_html5_video()
        with open("myvideo.html","w") as f:
            print(anim_c3.to_html5_video(), file=f)

        HtmlFile = open("myvideo.html", "r")
        #HtmlFile="myvideo.html"
        source_code = HtmlFile.read() 
        components.html(source_code, height = 900,width=900)

        st.title("Patients")
        #fig1, anim_c3= evoked.animate_topomap('eeg', times=times1, blit=False, frame_rate=1)
        fig1, anim_c4= evoked_p.animate_topomap('eeg', times=times1, blit=False, frame_rate=1)
        #st.video(anim_c3)
        anim_c4.save(r'Animation2.mp4')
        #HtmlFile = line_ani.to_html5_video()
        with open("myvideo.html","w") as f:
            print(anim_c4.to_html5_video(), file=f)

        HtmlFile = open("myvideo.html", "r")
        #HtmlFile="myvideo.html"
        source_code = HtmlFile.read() 
        components.html(source_code, height = 900,width=900)


    if task=="Topographic Maps":
        col1, col2 = st.columns([1,1])

        with col1:
             but1 = st.button('Patient', key="1")
        with col2:
             but2 = st.button('Healthy', key="2")
          #wargs={vmin= -5.0, vmax=5}
        if but1 is True:
            evoked_p.plot_topomap( cmap='Spectral_r', res=32,
                            outlines='skirt', contours=4, )
            times=[0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050 ]
            #fig2 = plt.figure(figsize=(6, 2))
            fig3=evoked_p.plot_topomap(times=times, ch_type='eeg', cmap='Spectral_r', res=32,
                            outlines='skirt', contours=4,)
            st.pyplot(fig3)
            #fig3 = plt.figure(figsize=(6, 2))
            times=[0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.1]
            fig4=evoked_p.plot_topomap(times=times, ch_type='eeg', cmap='Spectral_r', res=32,
                            outlines='skirt', contours=4,)
            st.pyplot(fig4)
            times=[0.11, 0.12, 0.013, 0.14, 0.15, 0.16, 0.17, 0.18, 0.2, 0.25, 0.3]
            fig5=evoked_p.plot_topomap(times=times, ch_type='eeg', cmap='Spectral_r', res=32,
                            outlines='skirt', contours=4,)
            st.pyplot(fig5)

        if but2 is True:
            evoked_h.plot_topomap( cmap='Spectral_r', res=32,
                            outlines='skirt', contours=4, )
            times=[0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050 ]
            #fig2 = plt.figure(figsize=(6, 2))
            fig3=evoked_h.plot_topomap(times=times, ch_type='eeg', cmap='Spectral_r', res=32,
                            outlines='skirt', contours=4,)
            st.pyplot(fig3)
            #fig3 = plt.figure(figsize=(6, 2))
            times=[0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.1]
            fig4=evoked_h.plot_topomap(times=times, ch_type='eeg', cmap='Spectral_r', res=32,
                            outlines='skirt', contours=4,)
            st.pyplot(fig4)
            times=[0.11, 0.12, 0.013, 0.14, 0.15, 0.16, 0.17, 0.18, 0.2, 0.25, 0.3]
            fig5=evoked_h.plot_topomap(times=times, ch_type='eeg', cmap='Spectral_r', res=32,
                            outlines='skirt', contours=4,)
            st.pyplot(fig5)

    if task =="Analyze Specific Electrodes": 
        col1, col2 = st.columns([1,1])

        with col1:
             but1 = st.button('Patient', key="1")
        with col2:
             but2 = st.button('Healthy', key="2")
          #wargs={vmin= -5.0, vmax=5}
        if but1 is True:
            #st.write("FCz") 
            fig11=evoked_p.plot_image(picks=[ 'FC1'])
            st.write("FC1") 
            st.pyplot (fig11)

            fig112=evoked_p.plot_image(picks=[ 'FCz'])
            st.write("FCz") 
            st.pyplot (fig112)

            fig113=evoked_p.plot_image(picks=[ 'FC2'])
            st.write("FC2") 
            st.pyplot (fig113)

            fig114=evoked_p.plot_image(picks=[ 'C3'])
            st.write("C3") 
            st.pyplot (fig114)

            fig115=evoked_p.plot_image(picks=[ 'C4'])
            st.write("C4") 
            st.pyplot (fig115)

            fig116=evoked_p.plot_image(picks=[ 'Cz'])
            st.write("Cz") 
            st.pyplot (fig116)

            fig117=evoked_p.plot_image(picks=[ 'P3'])
            st.write("P3") 
            st.pyplot (fig117)
            fig118=evoked_p.plot_image(picks=[ 'P4'])
            st.write("P4") 
            st.pyplot (fig118)

            fig119=evoked_p.plot_image(picks=[ 'Pz'])
            st.write("Pz") 
            st.pyplot (fig119)

        if but2 is True:
            #st.write("FCz") 
            fig11=evoked_h.plot_image(picks=[ 'FC1'])
            st.write("FC1") 
            st.pyplot (fig11)

            fig112=evoked_h.plot_image(picks=[ 'FCz'])
            st.write("FCz") 
            st.pyplot (fig112)

            fig113=evoked_h.plot_image(picks=[ 'FC2'])
            st.write("FC2") 
            st.pyplot (fig113)

            fig114=evoked_h.plot_image(picks=[ 'C3'])
            st.write("C3") 
            st.pyplot (fig114)

            fig115=evoked_h.plot_image(picks=[ 'C4'])
            st.write("C4") 
            st.pyplot (fig115)

            fig116=evoked_h.plot_image(picks=[ 'Cz'])
            st.write("Cz") 
            st.pyplot (fig116)

            fig117=evoked_h.plot_image(picks=[ 'P3'])
            st.write("P3") 
            st.pyplot (fig117)
            fig118=evoked_h.plot_image(picks=[ 'P4'])
            st.write("P4") 
            st.pyplot (fig118)

            fig119=evoked_h.plot_image(picks=[ 'Pz'])
            st.write("Pz") 
            st.pyplot (fig119)

    if task =="Temporal Analysis": 
        st.title("Patients")
        fig21=evoked_p.plot(picks='eeg', spatial_colors=True, gfp=True)
        st.pyplot(fig21)
        st.title("Healthy")

        fig22=evoked_h.plot(picks='eeg', spatial_colors=True, gfp=True)
        st.pyplot(fig22)
        st.title("Patients - Peaks")
        fig223=evoked_p.plot_joint(times='peaks')
        st.pyplot(fig223)
        st.title("Healthy - Peaks")
        fig224=evoked_h.plot_joint(times='peaks')
        st.pyplot(fig224)
