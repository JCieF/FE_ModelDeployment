import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
import os

# Verify model file exists
if not os.path.exists('finalExam.h5'):
    st.error("Model file 'finalExam.h5' not found. Please check the file path.")
else:
    try:
        # Load the model
        model = load_model('finalExam.h5')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise

# Function to preprocess the URL (assuming your model requires some preprocessing)
def preprocess_url(url):
    # Implement your preprocessing steps here
    # This is just a placeholder example
    return np.array([len(url)])  # Example: using the length of the URL as a feature

# Define your app
def main():
    st.title("URL Legitimacy Checker")
    url_input = st.text_input("Enter a URL:")
    if st.button("Check URL"):
        if url_input:
            try:
                processed_input = preprocess_url(url_input)
                prediction = model.predict(processed_input.reshape(1, -1))
                st.write(f"Prediction: {'Legit' if prediction[0] > 0.5 else 'Not Legit'}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.error("Please enter a URL.")

if __name__ == '__main__':
    main()
