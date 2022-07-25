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
        
    # load style, get state (usr and pw)
local_css("style.css")

#state = _get_state()

image = Image.open('logo-w.png')

st.sidebar.image(image, use_column_width=True)


task = st.selectbox('Select Task', ['', "Classification of MS Patients vs Healthy"])

if task =="Classification of MS Patients vs Healthy":
    from PIL import Image
    image = Image.open('classification.png')
    st.title("Classification Accuracy: 83.40") 
    st.image(image)
    
