import warnings
warnings.filterwarnings("ignore")
import sys, os, os.path
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
from pylab import rcParams
import streamlit as st
from google.cloud import storage
st.set_option('deprecation.showPyplotGlobalUse', False)
rcParams['figure.figsize'] = 12, 8


def file_loader(loaded_file1=None, loaded_file2=None):

    st.write("""
    # **MyelinH Parkinson Analysis Interface**  
    This system performs early diagnosis of parkinson's disease and monitoring of the patient health status.

    """)

    service_account = 'myelinh-gcloud-303512-da6502d6a766.json'
    client = storage.Client.from_service_account_json(service_account)

    data_bucket = client.get_bucket('myelinh-gcloud-303512.appspot.com')

    blobs = client.list_blobs(data_bucket, prefix='data-parkinsons/deep_analysis')
    l_blobs = [str(blob.name) for blob in blobs]
    l_blobs[0] = None

    if loaded_file1 is not None and loaded_file2 is not None:
        st.write('Files loaded in. you may skip loading files')
        uploaded_file_string1 = st.selectbox("Choose a different Test Patient", l_blobs, key=70)
        uploaded_file_string2 = st.selectbox("Choose a different Control Patient", l_blobs, key=90)

        if uploaded_file_string1 and uploaded_file_string2 is not None:
            if st.button("Upload new Files"):
                uploaded_file1 = data_bucket.get_blob(uploaded_file_string1)
                uploaded_file2 = data_bucket.get_blob(uploaded_file_string2)
                uploaded_file1.download_to_filename('file.csv', raw_download=True)
                uploaded_file2.download_to_filename('file2.csv', raw_download=True)

                data1 = np.loadtxt('file.csv', delimiter=',')
                data2 = np.loadtxt('file2.csv', delimiter=',')

                return data1, data2

    else:
        # Collects user input features into dataframe
        blobs = client.list_blobs(data_bucket, prefix='data-parkinsons/deep_analysis')
        l_blobs = [str(blob.name) for blob in blobs]
        l_blobs[0] = None

        uploaded_file_string1 = st.selectbox("Choose a Test Patient", l_blobs, key=70)
        uploaded_file_string2 = st.selectbox("Choose a Control Patient", l_blobs, key=90)

        if uploaded_file_string1 and uploaded_file_string2 is not None:
            uploaded_file1 = data_bucket.get_blob(uploaded_file_string1)
            uploaded_file2 = data_bucket.get_blob(uploaded_file_string2)
            uploaded_file1.download_to_filename('file.csv', raw_download=True)
            uploaded_file2.download_to_filename('file2.csv', raw_download=True)

            data1 = np.loadtxt('file.csv', delimiter=',')
            data2 = np.loadtxt('file2.csv', delimiter=',')

            return data1, data2

    data1 = loaded_file1
    data2 = loaded_file2
    return data1, data2


def create_MNE(patient, control, channels):
    # Create MNE Object
    raw_data_patient, info_patient, events_patient, event_dict = \
        BodyMirror.EEG_deep_analysis.convert_mne(channels, patient, type_id='patient', type_value=0, window_size=12,
                                                 crop_factor=5, montage='standard_1020')

    raw_data_control, info_control, events_control, event_dict_control = BodyMirror.EEG_deep_analysis.convert_mne(
        channels, control, type_id='control', type_value=0, window_size=12, crop_factor=5,
        montage='standard_1020')

    if 1:
        raw_data_patient = raw_data_patient.filter(l_freq=0.1, h_freq=None)
        raw_data_patient = raw_data_patient.filter(l_freq=2, h_freq=70)
        raw_data_patient = raw_data_patient.notch_filter(np.arange(60, 241, 60))

    if 1:
        raw_data_control = raw_data_control.filter(l_freq=0.1, h_freq=None)
        raw_data_control = raw_data_control.filter(l_freq=2, h_freq=70)
        raw_data_control = raw_data_control.notch_filter(np.arange(60, 241, 60))

    return raw_data_patient, raw_data_control, events_patient, event_dict


def visualization(loaded_file1, loaded_file2):

    fig = None
    output = None

    lowcut = 2
    highcut = 70

    # notch

    f0 = 60
    Q = 30

    channels = np.load('channels.npy')

    # Filter using Bodymirror
    lowcut = 2
    highcut = 70
    # notch
    f0 = 60
    Q = 30
    patient0 = loaded_file1
    if 0:
        patient0 = BodyMirror.signal.butter_highpass(loaded_file1, cutoff=0.1)
        patient0 = BodyMirror.signal.butter_bandpass(patient0, lo=lowcut, hi=highcut)
        patient0 = BodyMirror.signal.notch(patient0, cutoff=f0, Q=Q)

    # show ICA results
    raw_data_p, raw_data_c,events_p, events_dict_p = create_MNE(patient0, loaded_file2, channels=channels)
    st.write("Perform ICA")
    channels = int(st.text_input("select number of channels to analyze"))
    if channels is None:
        st.write("enter a valid number")
    else:
        ica = BodyMirror.EEG_deep_analysis.apply_ica_mne(data=raw_data_p, info=raw_data_p.info,
                                                     n_components=channels)
    if st.button('Show ICA'):
        st.pyplot(ica.plot_sources(raw_data_p), use_container_width=True)

    if st.button("Show Topographic maps"):
        reconst_raw_s1 = BodyMirror.EEG_deep_analysis.correct_data_ica(ica, raw_data_p, exclude=[0, 12])

        epochs, evoked = BodyMirror.EEG_deep_analysis.topographic_plot(tmin=0.0, tmax=5.0, data=reconst_raw_s1,
                                                                       events=events_p, event_dict=events_dict_p,
                                                                       montage='standard_1020', time="peaks",
                                                                       rejection=False)

        ts_args = dict(gfp=True, spatial_colors=False)
        topomap_args = dict(sensors=True)

        fig_1_1 = evoked.plot_joint(title='Parkisnon Patient Data', times='peaks',
                                    ts_args=ts_args, topomap_args=topomap_args)

        st.write(fig_1_1)

    if st.button("Plot Power Density"):
        reconst_raw_s1 = BodyMirror.EEG_deep_analysis.correct_data_ica(ica, raw_data_p, exclude=[0, 12])
        st.pyplot(reconst_raw_s1.plot_psd(fmax=50))
    if st.button("Plot Deep Analysis"):
        reconst_raw_s1 = BodyMirror.EEG_deep_analysis.correct_data_ica(ica, raw_data_p, exclude=[0, 12])

        epochs, evoked = BodyMirror.EEG_deep_analysis.topographic_plot(tmin=0.0, tmax=5.0, data=reconst_raw_s1,
                                                                       events=events_p, event_dict=events_dict_p,
                                                                       montage='standard_1020', time="peaks",
                                                                       rejection=False)
        fig = BodyMirror.EEG_deep_analysis.topo_deep_anaylsis(epochs)
        st.pyplot(fig, use_container_width=True)

    if st.button("Perform Source Localization"):
        reconst_raw_s1 = BodyMirror.EEG_deep_analysis.correct_data_ica(ica, raw_data_p, exclude=[0, 12])
        epochs, evoked = BodyMirror.EEG_deep_analysis.topographic_plot(tmin=0.0, tmax=5.0, data=reconst_raw_s1,
                                                                       events=events_p, event_dict=events_dict_p,
                                                                       montage='standard_1020', time="peaks",
                                                                       rejection=False)

        # source localization
        subjects_dir = 'freesurfer/subjects/'
        subject = 'daniel'
        bem_dir = op.join(subjects_dir, subject, 'bem')
        mne.utils.set_config("SUBJECTS_DIR", subjects_dir, set_env=True)
        # Read localization files

        src_data = mne.read_source_spaces('freesurfer/subjects/src')
        bem_sol = mne.read_bem_solution('freesurfer/subjects/daniel/daniel-5120-5120-5120-bem-sol.fif')
        trans = 'freesurfer/subjects/daniel/daniel-trans-z-trans.fif'

        # Forward model

        fwd = make_forward_solution(evoked.info, trans, src_data, bem_sol)

        # Cov matrix

        cov_p = mne.compute_covariance(epochs, method='auto')

        # Compute inverse solution

        inv_p = make_inverse_operator(evoked.info, fwd, cov_p, loose=0.2, depth=0.8)

        # Source loc

        stc_p2 = BodyMirror.EEG_deep_analysis.source_localization(evoked, inverse=inv_p, subjects_dir=subjects_dir,
                                                                  method="dSPM", split=False)

        from mne.viz import plot_alignment
        from mayavi import mlab
        fig = plot_alignment(raw_data_patient.info, trans, subject='daniel', dig=False,
                             eeg=['original', 'projected'], meg=[],
                             coord_frame='head', subjects_dir=subjects_dir)
        mlab.view(135, 80)

        st.write(fig)




