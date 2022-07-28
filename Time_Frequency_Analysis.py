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

#import myelinh_functions
#import text_to_speech


from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from mne.datasets import somato

def time_frequency(epochs_p):

    freqs = np.logspace(*np.log10([6, 35]), num=8)
    n_cycles = freqs / 2.  # different number of cycle per frequency
    power_p, itc_p = tfr_morlet(epochs_p, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=True, decim=3, n_jobs=1)


    task5 = st.selectbox('Select Task', ['', "Time Frequency Analysis"],  key="222456986")

    if task5=="Time Frequency Analysis":

        fig7=power_p.plot_joint(baseline=(-1.3, 0), mode='mean', tmin=-.9, tmax=1,
                         timefreqs=[(.1, 10)])
        st.pyplot(fig7)

        st.pyplot(itc_p.plot_topo(title='Inter-Trial coherence', vmin=0., vmax=1., cmap='Reds'))
        
def time_frequency_h(epochs_p):

    freqs = np.logspace(*np.log10([6, 35]), num=8)
    n_cycles = freqs / 2.  # different number of cycle per frequency
    power_p, itc_p = tfr_morlet(epochs_p, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=True, decim=3, n_jobs=1)


    task6 = st.selectbox('Select Task', ['', "Time Frequency Analysis"], key="2224569864563")

    if task6=="Time Frequency Analysis":

        fig7=power_p.plot_joint(baseline=(-1.3, 0), mode='mean', tmin=-.9, tmax=1,
                         timefreqs=[(.1, 10)])
        st.pyplot(fig7)

        st.pyplot(itc_p.plot_topo(title='Inter-Trial coherence', vmin=0., vmax=1., cmap='Reds'))
        






