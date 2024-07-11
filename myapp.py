import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import traceback

try:
    # Load the model
    model = load_model('finalExam.h5')
    st.write("Model loaded successfully.")
    
    # Function to preprocess the URL (assuming your model requires some preprocessing)
    def preprocess_url(url):
        # Implement your preprocessing steps here
        # This is just a placeholder example
        return np.array([len(url)])  # Example: using the length of the URL as a feature

    # Define the Streamlit app
    st.title("URL Legitimacy Checker")
    url_input = st.text_input("Enter a URL:")
    
    if st.button("Check URL"):
        if url_input:
            try:
                # Preprocess the input URL
                processed_input = preprocess_url(url_input)
                # Make the prediction
                prediction = model.predict(processed_input.reshape(1, -1))
                # Display the result
                st.write(f"Prediction: {'Legit' if prediction[0] > 0.5 else 'Not Legit'}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Please enter a URL.")
    
except Exception as e:
    st.error("Error loading model:")
    st.text(str(e))
    st.text(traceback.format_exc())
