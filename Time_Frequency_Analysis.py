import streamlit as st
#from streamlit.hashing import _CodeHasher
from streamlit.legacy_caching.hashing import _CodeHasher



import warnings
import matplotlib.pyplot as plt

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

import numpy as np

import joblib
import streamlit as st
import pandas as pd
import numpy as np

import pickle
from sklearn.ensemble import RandomForestClassifier
import scipy.linalg as la
from io import BytesIO

import sys, os, os.path


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
        





from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from mne.datasets import somato
freqs = np.logspace(*np.log10([6, 35]), num=8)
n_cycles = freqs / 2.  # different number of cycle per frequency
power_p, itc_p = tfr_morlet(epochs_p, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)

from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from mne.datasets import somato
freqs = np.logspace(*np.log10([6, 35]), num=8)
n_cycles = freqs / 2.  # different number of cycle per frequency
power_h, itc_h = tfr_morlet(epochs_h, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)

task = st.selectbox('Select Task', ['', "Time Frequency Analysis", "Comparison of Patients & Healthy Brain Responses"])

if task=="Time Frequency Analysis":

    st.title("Patients")
    fig7=power_p.plot_joint(baseline=(-1.3, 0), mode='mean', tmin=-.9, tmax=1,
                     timefreqs=[(.1, 10)])
    st.pyplot(fig7)

    st.pyplot(itc_p.plot_topo(title='Inter-Trial coherence', vmin=0., vmax=1., cmap='Reds'))

    st.title("Healthy")
    fig8=power_h.plot_joint(baseline=(-1.3, 0), mode='mean', tmin=-.9, tmax=1,
                     timefreqs=[(.1, 10)])
    st.pyplot(fig8)
    st.pyplot(itc_h.plot_topo(title='Inter-Trial coherence', vmin=0., vmax=1., cmap='Reds'))
    
evokeds_list= [evoked_h, evoked_p]
conds = ('Healthy', 'MS_patient')
evks = dict(zip(conds, evokeds_list))

def custom_func(x):
    return x.max(axis=1)



if task=="Comparison of Patients & Healthy Brain Responses":
    
 
    for combine in ('mean', 'median', 'gfp', custom_func):
        fig=mne.viz.plot_compare_evokeds(evks, picks='eeg', combine=combine)


