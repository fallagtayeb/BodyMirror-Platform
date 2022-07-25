import streamlit as st

def app():
    Admin = 'Admin'
    A_Pwd = 'admin'
    st.title('login')
    st.write('login')
    User = st.text_input('Username: ')
    Pwd = st.text_input('Password: ')

    if User == Admin and Pwd == A_Pwd:
        st.write('Successfull login')
        return
    else:
        st.write('Incorrect username/pwd')
