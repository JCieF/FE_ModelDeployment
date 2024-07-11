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
    max_sequence_length = 100  # Example: maximum sequence length expected by your model
    vocab_size = 10000  # Example: vocabulary size
    
    # Tokenization (example using simple word splitting)
    tokens = text.split()
    
    # Limit tokens to max_sequence_length
    tokens = tokens[:max_sequence_length]
    
    # Convert tokens to indices or use other methods for encoding
    # Example: use keras.preprocessing.text.Tokenizer for tokenization
    
    # Example: pad_sequences for padding sequences to a fixed length
    padded_tokens = pad_sequences([tokens], maxlen=max_sequence_length, padding='post')
    
    return padded_tokens

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
