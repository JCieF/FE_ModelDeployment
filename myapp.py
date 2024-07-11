import streamlit as st
from tensorflow.keras.models import load_model
import traceback

try:
    model = load_model('finalExam.h5')
    st.write("Model loaded successfully.")
except Exception as e:
    st.error("Error loading model:")
    st.text(str(e))
    st.text(traceback.format_exc())
