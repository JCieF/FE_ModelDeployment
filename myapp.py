import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the Keras model
model_path = 'finalExam.keras'
model = load_model(model_path)

# Function to preprocess user input
def preprocess_input(text):
    # Implement your preprocessing logic here
    # Example: Tokenization, padding for sequences
    return np.array([text])  # Example: return a numpy array

# Define your Streamlit app
def main():
    st.title('Text Classification App')
    user_input = st.text_input('Enter text:')
    
    if st.button('Predict'):
        if user_input.strip():  # Check if input is not empty
            try:
                # Preprocess user input
                processed_input = preprocess_input(user_input)
                
                # Make prediction
                prediction = model.predict(processed_input)
                
                st.write(f'Prediction: {prediction[0]}')
                
            except Exception as e:
                st.error(f'Error: {e}')
        else:
            st.warning('Please enter some text.')

# Run the app
if __name__ == '__main__':
    main()
