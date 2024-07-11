import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load the model
model = load_model('finalExam.keras')

# Define your app
def main():
    st.title("My Streamlit App")
    user_input = st.text_input("Enter some data:")
    if st.button("Predict"):
        # Convert the user input to the format your model expects
        data = np.array([float(x) for x in user_input.split(",")])
        prediction = model.predict(data.reshape(1, -1))
        st.write(f"Prediction: {prediction[0]}")

if __name__ == '__main__':
    main()
