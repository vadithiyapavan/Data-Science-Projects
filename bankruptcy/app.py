import numpy as np
import pickle
import streamlit as st

# Load the pre-trained model
pickle_in = open("knn.pkl", "rb")
classifier = pickle.load(pickle_in)

def welcome():
    return "Welcome to the Bankruptcy Detection Model!"

def predict_bankruptcy(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk):
    # Prepare the input feature vector
    features = [[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk]]
    
    # Make prediction using the trained model
    prediction = classifier.predict(features)
    
    # Check the result: Model might return a numeric or string label, so process it
    return "Bankrupt" if prediction == 1 else "Not Bankrupt"

def main():
    # App title
    st.title("Bankruptcy Detector")
    
    # Styling for header
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bankruptcy Detection ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Input fields for the user to enter data
    industrial_risk = st.number_input("Industrial Risk (0-1)", min_value=0.0, max_value=1.0, value=0.0)
    management_risk = st.number_input("Management Risk (0-1)", min_value=0.0, max_value=1.0, value=0.0)
    financial_flexibility = st.number_input("Financial Flexibility (0-1)", min_value=0.0, max_value=1.0, value=0.0)
    credibility = st.number_input("Credibility (0-1)", min_value=0.0, max_value=1.0, value=0.0)
    competitiveness = st.number_input("Competitiveness (0-1)", min_value=0.0, max_value=1.0, value=0.0)
    operating_risk = st.number_input("Operating Risk (0-1)", min_value=0.0, max_value=1.0, value=0.0)

    # Prediction output initialization
    result = ""
    
    # Prediction button
    if st.button("Predict"):
        result = predict_bankruptcy(industrial_risk, management_risk, financial_flexibility, credibility, competitiveness, operating_risk)
        
    # Display result
    st.success(f"The output is: {result}")

    # About section
    if st.button("About"):
        st.text("This app helps detect bankruptcy using a Machine Learning model.")
        st.text("Built with Streamlit and trained using KNN algorithm.")

if __name__ == '__main__':
    main()
