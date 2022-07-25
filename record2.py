#import pygds as g
import os
import warnings
warnings.filterwarnings("ignore")
import sys, os, os.path

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
#from firebase import firebase
from getpass import getpass
from time import sleep
import streamlit as st
import codecs, json
from stqdm import stqdm
import text_to_speech


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

text1= "Welcome to the visual evoked potential remote examination. Please stay still, focus on the red square on the center of the screen, and try not to blink. This recording will run for 30 seconds."

def record_vep_data (times=30):
    #text_to_speech.voice_gen(text1)
    date = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    filename=f"filename_{date}"
   
    #d = g.GDS()
    st.video("testMovie5.mp4", format="video/mp4", start_time=0)
    #minf_s = sorted(d.GetSupportedSamplingRates()[0].items())[0]
    #d.SamplingRate, d.NumberOfScans = minf_s
    #for ch in d.Channels:
     #   ch.Acquire = True
    #d.SetConfiguration()
    s=(20,20)
    data =np.ones(s) 
    #d.GetData(times*d.SamplingRate)
    sampling_rate = 250
    #d.SamplingRate
    #channels = d.GetChannelNames()
    channels=['Fp1', "Fp2", "AF3", "AF4", "F7", "F3","Fz", "F4", "F8", "FC5", "FC1",
          "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6","P7", "P3"
          , "Pz", "P4", "P8", "PO7", "PO3", "PO4", "PO8", "Oz"]
    #d.Close()
    #del d

    for _ in stqdm(range(times)):
        sleep(0.5)
    if not data.size:
        st.write('Unsuccessful brain monitoring session. Please try again.')
    else:

        #db=firebase.database()
        #data = data.tolist() # nested lists with same data, indices
        #json_data = json.dumps(data, indent = 4)

        #user_data={"user": user_name, "Connection_time":connection_time, "data_type":"EEG", "channels":channels, "recording_info":filename, "sampling_rate":sampling_rate, "data": json_data, "experiment":"vep"}
        #db.child("bodymirror-users").push(data)   
        #db.child("bodymirror-users").child(filename).set(user_data)
        for _ in stqdm(range(5)):
                sleep(0.5)
        st.write('A successful visual evoked potential monitoring session has been performed.')
        return data, filename, sampling_rate, channels
    
    



    
text2="A remote neuro exam is about to start. Please listen carefully to the provided instructions in each recording session:"

text3= "A motor function game is about to start. This game is designed to assess motor functions as well as to be used for rehabilitation. Please focus on the screen and try to passively imagine moving your right hand and fingers in the same way as shown on the screen. Please only perform an imagination of the movement and do not actively move your hands. This recording will start now and last for 1 minute." 

text4=" A Visual spatial attention exam is about to start. Please follow the provided instructions on the screen. This game is designed to assess your attention and cognitive abilities as well as for neurofeedback."

text5="A remote motor & coordination neuro exam is about to start. Please listen carefully to the following instructions. Please imagine a reach movement by bringing the cursor (square) toward one of the six center-out target locations (up-left, up-right, center-left, center-right, down-left, down-right). Once the square hits the target and turns red, please imagine performing a grasping movement on that specific target. This recording will start now and last for 1 minute."

def record_data():
    col1, col2, col3 = st.columns([1,1,1])
    col4, col5, col6 = st.columns([2,2,2])

    with col1:
         but1 = st.button('Remote Neurological Examination', key="1")
    with col2:
         but2 = st.button('Neurofeedback & Cognitive Assessment', key="2")
    with col3:
         but3 = st.button('Neurorehabilitation & Motor Function Assessment', key="3")
    with col4:
        but4 = st.button('Motor & Coordination Function Assessment', key="4")
    with col5:
        but5 = st.button('Coordination Function Assessment', key="5")
    with col6:
        but6 = st.button('Alpha Brain Wave Assessment', key="6")
   
    
    if but1 is True:
        #text_to_speech.voice_gen(text2)
        #myobj = gTTS(text=mytext, lang=language, slow=False)
        data, filename, sampling_rate, channels=record_vep_data()
        #import rest
        #rest.run()
    elif but2 is True: 
        #st.audio("vhand.mp4", format="video/mp4", start_time=0)
        import subprocess
        #text_to_speech.voice_gen(text4)
        #import spatial_cueing_paradigm
        subprocess.call(['python', "./spatial_cueing_paradigm.py"])
        #p=subprocess.Popen(['python', "./spatial_cueing_paradigm.py"])
        #p.terminate()
        but2=False
        #subprocess.call("./spatial_cueing_paradigm.py", shell=True)
        #exec(open("spatial_cueing_paradigm.py").read())
    elif but3 is True:
        #import subprocess
        #text_to_speech.voice_gen(text3)
        st.video("vhand.mp4", format="video/mp4", start_time=0)
        for _ in stqdm(range(10)):
                sleep(5)
        st.write('A successful motor function assessment session has been performed.')
        #st.video("video_virtual_hand.avi", start_time=0)
        #vhand()
        #subprocess.call(['python', "./experimental_paradigms/EMG/complex_movements/offline/EMG_run_session_offline.py"])
        #C:\Users\55145178\Desktop\Brain\patient_app\experimental_paradigms\EMG\complex_movements\offline
        #import rest
        #rest.run()
    elif but4 is True:
        #text_to_speech.voice_gen(text5)
        import subprocess
        #import reach_to_grasp
        #reach_to_grasp.run_session()
        subprocess.call(['python', "./reach_to_grasp.py"])
        #run([sys.executable, './experimental_paradigms/ReachToGrasp/src/reach_to_grasp.py'])
        #subprocess.call(['python', "./reach_to_grasp.py"])
        #p=subprocess.Popen(['python', "./spatial_cueing_paradigm.py"])
        #p.terminate()
        but4=False
    elif but5 is True:
        #text_to_speech.voice_gen(text5)
        st.video("vreach.mp4", format="video/mp4", start_time=0)
        
        
    
        
       
        
    else:
        data=None
        filename=None
        sampling_rate=None
        channels=None
    

    
