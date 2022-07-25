import pygds as g
import os
import warnings
warnings.filterwarnings("ignore")
import sys, os, os.path
sys.path.append('/BodyMirror')
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
import pyrebase
import json
from firebase import firebase
from getpass import getpass
from time import sleep
import streamlit as st
import codecs, json
from stqdm import stqdm


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

firebase = pyrebase.initialize_app(config)
def record_vep_data (times=30):
    but1 = st.button('Remote Neuro Exam', key="1")
    date = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    filename=f"filename_{date}"
    if but1 is True:
        d = g.GDS()
        st.video("testMovie5.mp4", format="video/mp4", start_time=0)
        minf_s = sorted(d.GetSupportedSamplingRates()[0].items())[0]
        d.SamplingRate, d.NumberOfScans = minf_s
        for ch in d.Channels:
            ch.Acquire = True
        d.SetConfiguration()
        data = d.GetData(times*d.SamplingRate)
        sampling_rate = d.SamplingRate
        #channels = d.GetChannelNames()
        channels=['Fp1', "Fp2", "AF3", "AF4", "F7", "F3","Fz", "F4", "F8", "FC5", "FC1",
              "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6","P7", "P3"
              , "Pz", "P4", "P8", "PO7", "PO3", "PO4", "PO8", "Oz"]
        d.Close()
        del d

        for _ in stqdm(range(times)):
            sleep(0.5)
        if not data.size:
            st.write('Unsuccessful brain monitoring session. Please try again.')
        else:
            
            db=firebase.database()
            data = data.tolist() # nested lists with same data, indices
            json_data = json.dumps(data, indent = 4)
            user_info = db.child("bodymirror-users").child("connected-users").get()
            user_name=user_info.each()[2].val()
            connection_time=user_info.each()[0].val()
            user_data={"user": user_name, "Connection_time":connection_time, "data_type":"EEG", "channels":channels, "recording_info":filename, "sampling_rate":sampling_rate, "data": json_data, "experiment":"vep"}
            #db.child("bodymirror-users").push(data)   
            db.child("bodymirror-users").child(filename).set(user_data)
            for _ in stqdm(range(5)):
                    sleep(0.5)
            st.write('A successful visual evoked potential monitoring session has been performed.')

    else:
        data=None
        filename=None
        sampling_rate=None
        channels=None
    return data, filename, sampling_rate, channels

    