import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model('model.h5')

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
        processed_input = preprocess_url(url_input)
        prediction = model.predict(processed_input.reshape(1, -1))
        st.write(f"Prediction: {'Legit' if prediction[0] > 0.5 else 'Not Legit'}")

if __name__ == '__main__':
    main()
