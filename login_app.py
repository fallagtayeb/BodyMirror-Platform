# Get a reference to the auth service
import streamlit as st

import json

from  streamlit.legacy_caching.hashing import _CodeHasher
#import myelinh_functions
with open('myelin-h-authentification-firebase-adminsdk-1eqje-c91381fc52.json') as json_file:
    data_j = json.load(json_file)
from PIL import Image
from datetime import datetime
import yaml
from streamlit_option_menu import option_menu

import streamlit_authenticator as stauth

import extra_streamlit_components as stx

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




def sign_up():
    try:
        if authenticator.register_user('Register user', preauthorization=True):
            st.success('User registered successfully')
    except Exception as e:
        st.error(e)

    with open('./config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)
        
    #input = st.empty()
    email=1
    password=1
    #email = st.text_input('Please sign up using your email address: \n', key="11")
    #password = st.text_input('Please enter a password password: \n', type="password")
    #but1 = st.button('sign up', key="1")
    #if but1 is True:
        #user = auth.create_user_with_email_and_password(email, password)
        #st.write('Thanks for signing up')
    return email, password



hashed_passwords = stauth.Hasher(['123']).generate()

with open('./config.yaml') as file:
    config = yaml.safe_load(file)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
#authentication_status = None
def login(authentication_status=None):
            
    #email = None
    
        
    #password = None
    #authentication_status=None
    #input = st.empty()
    #email = st.text_input('Please sign in with your email address: \n', key="22")
    #password = st.text_input('Please sign in with your password: \n', type="password")
    #but2 = st.button('login', key="2")
    #but3 = st.button('reset password', key="3")
    email, authentication_status, password = authenticator.login('Login', 'main')
    
    if authentication_status:
        #user = auth.sign_in_with_email_and_password(email, password)
        st.write('Thanks for signing in')
        authenticator.logout('Logout', 'main')
        
    elif authentication_status == False:
        st.error('Username/password is incorrect')
        
    elif authentication_status == None:
        st.warning('Please enter your username and password')
    else:
        authentication_status= None
        email=None
        password=None
        


    return authentication_status


def reset():
    input = st.empty()
    email = st.text_input('Please enter your email address: \n', key="33")
    but3 = st.button('reset', key="3")
    if but3 is True:
        auth.send_password_reset_email(email)
        st.write('An email to reset your password has been sent to you')
    return email

def authentification(data):


    st.title('Authentification')

    access = None
    license = st.text_input(' ', type="password")
    #Name = st.text_input('', key=1)
    #db=firebase.database()
    if license == data['private_key_id'] and access is None:
            import time
            st.success('Successful Authentification')
            time.sleep(1)
        
            #st.write('Successful Authentification')
            access=1
        
    else:
        access=None 
        st.warning('Please enter your License ID')
             

    


    

    return access






