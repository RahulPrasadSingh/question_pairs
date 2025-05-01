import streamlit as st
from utils import load_model, predict

# Load the trained model
model, vectorizer = load_model()

st.title("Machine Learning Model in App")

question1 = st.text_input("Enter Question 1")
question2 = st.text_input("Enter Question 2")

if st.button("Check Duplicate"):
    if question1 and question2:
        result = predict(question1, question2, model, vectorizer)
        if result == 1:
            st.success("These questions are duplicates!")
        else:
            st.error("These questions are not duplicates.")
    else:
        st.warning("Please enter both questions.")
