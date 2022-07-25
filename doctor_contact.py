#import pygds as g
import os
import numpy as np
from twilio.rest import Client
import streamlit as st
from trycourier import Courier
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt 
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
from datetime import datetime
import streamlit as st
import time
#import pyrebase
import json
#from firebase import firebase
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

#firebase = pyrebase.initialize_app(config)
#db=firebase.database()





def doctor_contact_func():
    button1=st.button('Contact Your Doctor')
    if button1==True:
        #user_info = db.child("bodymirror-users").child("connected-users").get()
        doctor_email="info@myelinh.com"
        #st.write(doctor_email)
        client = Courier(auth_token="pk_prod_KMJJ8SEJJ943DJJDZGS9YZK5D2ZG")
        resp = client.send_message(
          message={
            "to": {
              "email": doctor_email
            },
            "content": {
              "title": "Patient Support",
              "body": "Your patient needs your support, please connect to Myelin-H dashboard to see the updates"
            },
            "data":{
              "joke": ""
            }
          }
        )
        st.write("An email has been sent to your doctor")





    




