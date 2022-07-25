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

epochs_h=mne.read_epochs("epochs_h.fif")

epochs_p=mne.read_epochs("epochs_p.fif") 


evoked_h = epochs_h.average()
evoked_h.apply_proj()

evoked_p = epochs_p.average()
evoked_p.apply_proj()
def local_css(file_name):
    with open(file_name) as f:
        hide_streamlit_style ="""
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        </style>
        """
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        


def plot_frequency_topo(evoked, fmin, fmax, times, title, **kwargs):
    """plot topographic maps for different frequency bands 

    This function takes the ``mne object`` and plot topographic maps for a specific frequency band 

    Parameters
    ----------
    evoked:
    
    fmin : integer-- f_min of band 
    fmax: integer-- f_max of band
    time: list of time points to visualize (could be "peaks")
    title: character-- frequency band (e.g. "alpha")to visualize

    Returns
    -------
    topographic plots
   
    """
    evoked_copy=evoked.copy()
    evoked_copy= evoked_copy.filter(fmin, fmax, fir_design='firwin')
    fig= evoked_copy.plot_topomap(times=times, ch_type='eeg', title=title)
    return fig

    
task = st.selectbox('Select Task', ['', "Plot Spectral Topographic Maps of Brain Waves", "Power Spectral Density of Brain Waves"])

if task =="Plot Spectral Topographic Maps of Brain Waves": 
    col1, col2 = st.columns([1,1])

    with col1:
         but1 = st.button('Patient', key="1")
    with col2:
         but2 = st.button('Healthy', key="2")
      #wargs={vmin= -5.0, vmax=5}
    if but1 is True:
    #time="peaks"
        times=[0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050 ]
        times1=[0.060, 0.070, 0.080, 0.090, 0.1, 0.12, 0.15, 0.2, 0.25, 0.28]
        kwargs={}
        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(1, 4, fir_design='firwin')
        fig31= evoked_copy.plot_topomap(times=times, ch_type='eeg', title="Delta")
        st.pyplot(fig31)

        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(1, 4, fir_design='firwin')
        fig32= evoked_copy.plot_topomap(times=times1, ch_type='eeg', title="Delta")
        st.pyplot(fig32)

        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(4, 7, fir_design='firwin')
        fig31= evoked_copy.plot_topomap(times=times, ch_type='eeg', title="Theta")
        st.pyplot(fig31)

        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(4, 7, fir_design='firwin')
        fig33= evoked_copy.plot_topomap(times=times1, ch_type='eeg', title="Theta")
        st.pyplot(fig33)


        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(8, 12, fir_design='firwin')
        fig31= evoked_copy.plot_topomap(times=times, ch_type='eeg', title="Alpha")
        st.pyplot(fig31)

        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(8, 12, fir_design='firwin')
        fig33= evoked_copy.plot_topomap(times=times1, ch_type='eeg', title="Alpha")
        st.pyplot(fig33)


        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(13, 30, fir_design='firwin')
        fig31= evoked_copy.plot_topomap(times=times, ch_type='eeg', title="Beta")
        st.pyplot(fig31)

        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(13, 30, fir_design='firwin')
        fig33= evoked_copy.plot_topomap(times=times1, ch_type='eeg', title="Beta")
        st.pyplot(fig33)

        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(30, 70, fir_design='firwin')
        fig31= evoked_copy.plot_topomap(times=times, ch_type='eeg', title="Gamma")
        st.pyplot(fig31)

        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(30, 70, fir_design='firwin')
        fig33= evoked_copy.plot_topomap(times=times1, ch_type='eeg', title="Gamma")
        st.pyplot(fig33)
    
    if but2 is True:
    #time="peaks"
        times=[0.015, 0.020, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050 ]
        times1=[0.060, 0.070, 0.080, 0.090, 0.1, 0.12, 0.15, 0.2, 0.25, 0.28]
        kwargs={}
        evoked_copy=evoked_h.copy()
        evoked_copy= evoked_copy.filter(1, 4, fir_design='firwin')
        fig31= evoked_copy.plot_topomap(times=times, ch_type='eeg', title="Delta")
        st.pyplot(fig31)

        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(1, 4, fir_design='firwin')
        fig32= evoked_copy.plot_topomap(times=times1, ch_type='eeg', title="Delta")
        st.pyplot(fig32)

        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(4, 7, fir_design='firwin')
        fig31= evoked_copy.plot_topomap(times=times, ch_type='eeg', title="Theta")
        st.pyplot(fig31)

        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(4, 7, fir_design='firwin')
        fig33= evoked_copy.plot_topomap(times=times1, ch_type='eeg', title="Theta")
        st.pyplot(fig33)


        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(8, 12, fir_design='firwin')
        fig31= evoked_copy.plot_topomap(times=times, ch_type='eeg', title="Alpha")
        st.pyplot(fig31)

        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(8, 12, fir_design='firwin')
        fig33= evoked_copy.plot_topomap(times=times1, ch_type='eeg', title="Alpha")
        st.pyplot(fig33)


        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(13, 30, fir_design='firwin')
        fig31= evoked_copy.plot_topomap(times=times, ch_type='eeg', title="Beta")
        st.pyplot(fig31)

        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(13, 30, fir_design='firwin')
        fig33= evoked_copy.plot_topomap(times=times1, ch_type='eeg', title="Beta")
        st.pyplot(fig33)

        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(30, 70, fir_design='firwin')
        fig31= evoked_copy.plot_topomap(times=times, ch_type='eeg', title="Gamma")
        st.pyplot(fig31)

        evoked_copy=evoked_p.copy()
        evoked_copy= evoked_copy.filter(30, 70, fir_design='firwin')
        fig33= evoked_copy.plot_topomap(times=times1, ch_type='eeg', title="Gamma")
        st.pyplot(fig33)
        
if task=="Power Spectral Density of Brain Waves":
    st.title("Patients") 
    fig4=epochs_p.plot_psd_topomap(ch_type='eeg', normalize=True)
    st.pyplot(fig4)
    
    st.title("Healthy") 
    fig5=epochs_h.plot_psd_topomap(ch_type='eeg', normalize=True)
    st.pyplot(fig5)
