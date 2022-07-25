import pygds as g
import os
from ctypes import *
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
import pyrebase
import json
from firebase import firebase
from getpass import getpass
from time import sleep
import streamlit as st
import codecs, json
from stqdm import stqdm
from pandas import DataFrame
import numpy as np
from pandas import DataFrame
from psychopy import visual, core, event
from time import time, strftime, gmtime
from optparse import OptionParser
from pylsl import StreamInfo, StreamOutlet
import os


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

duration=120

firebase = pyrebase.initialize_app(config)
def record_data (times=duration):
    #but1 = st.button('Remote health monitoring session', key="1")
    date = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    filename=f"filename_{date}"
    if 1:
        d = g.GDS()
        minf_s = sorted(d.GetSupportedSamplingRates()[0].items())[0]
        d.SamplingRate, d.NumberOfScans = minf_s
        for ch in d.Channels:
            ch.Acquire = True
        d.SetConfiguration()
        data = d.GetData(times*d.SamplingRate)
        sampling_rate = d.SamplingRate
        channels = d.GetChannelNames()
        d.Close()
        del d
        np.save(filename, data)
        for _ in stqdm(range(times)):
            sleep(0.5)
        if not data.size:
            st.write('Unsuccessful brain monitoring session. Please try again.')
        else:
            
            db=firebase.database()
            data = data.tolist() # nested lists with same data, indices
            json_data = json.dumps(data, indent = 4)
            user_data={"user": "zied", "data_type":"EEG", "channels":channels, "recording_info":filename, "sampling_rate":sampling_rate, "data": json_data}
            #db.child("bodymirror-users").push(data)   
            db.child("bodymirror-users").child(filename).set(user_data)
            for _ in stqdm(range(5)):
                    sleep(0.5)
            st.write('A successful brain monitoring session. Your doctor has received your health status report and has been notified.')

    else:
        data=None
        filename=None
        sampling_rate=None
        channels=None
    return data, filename, sampling_rate, channels

def present(duration=120):

    # create
    #info = StreamInfo("Markers", "Markers", 1, 0, "int32", "myuidw43536")

    # next make an outlet
    #outlet = StreamOutlet(info)

    markernames = [1, 2]

    start = time()

    n_trials = 2000
    iti = 0.2
    jitter = 0.1
    soa = 0.2
    record_duration = np.float32(duration)

    # Setup log
    position = np.random.randint(0, 2, n_trials)
    trials = DataFrame(dict(position=position, timestamp=np.zeros(n_trials)))

    # graphics
    mywin = visual.Window(
        [1366, 768], monitor="testMonitor", units="deg", fullscr=True
    )
    grating = visual.GratingStim(win=mywin, mask="circle", size=20, sf=4)
    fixation = visual.GratingStim(win=mywin, size=0.2, pos=[0, 0], sf=0, rgb=[1, 0, 0])

    for ii, trial in trials.iterrows():
        # inter trial interval
        core.wait(iti + np.random.rand() * jitter)

        # onset
        grating.phase += np.random.rand()
        pos = trials["position"].iloc[ii]
        grating.pos = [25 * (pos - 0.5), 0]
        grating.draw()
        fixation.draw()
        #outlet.push_sample([markernames[pos]], time())
        mywin.flip()

        # offset
        core.wait(soa)
        fixation.draw()
        mywin.flip()
        if len(event.getKeys()) > 0 or (time() - start) > record_duration:
            break
        event.clearEvents()
    # Cleanup
    mywin.close()



from psychopy import visual, core, event
import time

# create window
# later adjust size to tablet!!!
win0 = visual.Window([1366, 768], monitor="testMonitor", units="deg", fullscr=True)

grating = visual.GratingStim(win=win0, mask="circle", size=20, sf=4)
fixation = visual.GratingStim(win=win0, size=0.2, pos=[0, 0], sf=0, rgb=[1, 0, 0])

win0.getMovieFrame(buffer='back')
startTime = time.time()
runTime = 12 # run stimulus for 15 seconds

n_trials = 2000
iti = 0.2
jitter = 0.1
soa = 0.2
duration=20
# Setup log
pos = np.random.randint(0, 2, n_trials)

while(time.time() - startTime < duration):
 #   win0.getMovieFrame(buffer='back') # get every upcoming frame?
    #gratingStimulus.setPhase(0.02, '+')
    # 1st parameter is for speed of drifting
    # 2nd parameter is for direction of drifint ('+': left to right)

    
    #core.wait(iti + np.random.rand() * jitter)
    grating.phase += np.random.rand()
    for i in range(30):
        win0.getMovieFrame()
        core.wait(iti + np.random.rand() * jitter)
        grating.pos = [25 * (pos[i] - 0.5), 0]
        grating.draw()
        fixation.draw()
        win0.flip()
        win0.getMovieFrame()

    if len(event.getKeys()) > 0:
        break
    event.clearEvents()
print("Play time: ", time.time()-startTime)    
win0.close()
win0.saveMovieFrames(fileName='testMovie5.mp4', fps=8) # save frames as video file

    