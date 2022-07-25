import streamlit as st
from PIL import Image
img= Image.open('logo-high-res.png')

st.set_page_config(page_title="BodyMirror-Platform", page_icon=img)



from  streamlit.legacy_caching.hashing import _CodeHasher

from streamlit.scriptrunner import get_script_run_ctx as get_report_ctx
import warnings
import matplotlib.pyplot as plt
#import seaborn as sns
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
#from google.cloud import storage
import pickle
from sklearn.ensemble import RandomForestClassifier
import scipy.linalg as la
from io import BytesIO
#import tensorflow as tf
#from tensorflow.python.lib.io import file_io
#import myelinh_functions
import login_app
#import visualization


import base64
import doctor_contact
#import vep
import yaml
from streamlit_option_menu import option_menu
#%matplotlib qt
import streamlit as st
import streamlit_authenticator as stauth
#import visualization
from PIL import Image
import base64
import json 

import extra_streamlit_components as stx

import read_data



config = {
  "apiKey": "AIzaSyCmlYv7hDaiandiCH1HAzmAvx7x4dsiu-c",
  "authDomain": "myelin-h-authentification.firebaseapp.com",
  "projectId": "myelin-h-authentification",
  "storageBucket": "myelin-h-authentification.appspot.com",
  "messagingSenderId": "207514364104",
  "appId": "1:207514364104:web:05ed56448c173645015dfd",
  "measurementId": "G-6ZMDGKPF5J",
  "databaseURL": "https://myelin-h-authentification-default-rtdb.europe-west1.firebasedatabase.app/"
}

#authentication_st
email = None
password = None
authentication_status = None
email1 = None
password1 = None
access = 1
login_state = 1

with open('myelin-h-authentification-firebase-adminsdk-1eqje-c91381fc52.json') as json_file:
    data_j = json.load(json_file)

try:
    from streamlit.scriptrunner import get_script_run_ctx
except ModuleNotFoundError:
    # streamlit < 1.8
    try:
        from streamlit.script_run_context import get_script_run_ctx  # type: ignore
    except ModuleNotFoundError:
        # streamlit < 1.4
        from streamlit.report_thread import (  # type: ignore
            get_report_ctx as get_script_run_ctx,
        )

            
from streamlit.server.server import Server
def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

#set_png_as_page_bg('2.png')

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "png"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-position: 275px 0px;
             background-size: auto;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )
    return 1

    # After Streamlit 0.65
    #from streamlit.report_thread import get_report_ctx
    #from streamlit.server.server import Server

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
        


@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-position: 275px 0px;
    background-size: auto;
 
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


#    background-position: 275px 0px;


def main():
   


    # load style, get state (usr and pw)
    local_css("style1.css")

    state = _get_state()

    image = Image.open('logo-w.png')

    st.sidebar.image(image, use_column_width=True)
    #set_png_as_page_bg('2.jpg')

    # Multipages

    pages = {
        "Authentification": authentification_m,
        "Clinical Data Analysis": functions,
        "Download Clinical Results": analysis,
    }
    
    
    v_menu=["Authentification", "Clinical Data Analysis", "Download Clinical Results" ]
    with st.sidebar:

                        #st.header("Data Analysis Options")

                        page = option_menu(
                            menu_title=None,  # required
                            options=v_menu,  # required
                            icons=None,  # optional
                            menu_icon="menu-down",  # optional
                            default_index=0,  # optional
                        )

    #page = st.sidebar.radio("Select your activity", tuple(pages.keys()))

    pages[page](state)

    # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    state.sync()


def authentification_m(state, data=data_j):
    if state.access_token is not None:
        #st.success('Successful Authentification')
        st.title("Welcome to Myelin-H Platform")
        mytext='Welcome to Myelin-H Platform'
        #text_to_speech.voice_gen(mytext)
        # load in variables
        state.data1 = None
        state.data2 = None
        if st.button("Log out"):
            
            state.clear()
    else:
        state.access_token = login_app.authentification(data)


def functions(state):
    if state.access_token == 1:
        read_data.data_read()
        #record2.record_data()
        #record.cloud_store(data, filename, sampling_rate, channels)
        #st.title("Welcome to MyelinH-1")
    else:
        st.error('Please enter your unique license ID')
        #st.write("Error, please login or sign up")


def analysis(state):
    if state.access_token == 1:
        #state.data1, state.data2 = visualization.file_loader()
        doctor_contact.doctor_contact_func()
        #emails=doctor_contact.email(button1)


    else:
        st.error('Please enter your unique license ID')
        #st.write("Error, please login or sign up")



# class session state creates a session which allows the dashboard to save variables while running processes without
# losing them each session, e.g. Username and password. This data is shared by all pages and allows multipaging on our
# website

class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            #"authentication_status": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun(None)
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun(None)

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)


def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


if __name__ == "__main__":
    main()

