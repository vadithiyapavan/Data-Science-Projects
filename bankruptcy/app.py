import numpy as np
import pickle
import streamlit as st

# Load the pre-trained model
import pickle
try:
    with open("model.pkl", "wb") as f:
        pickle.dump(classifier, f)
    print("Model saved successfully!")
except Exception as e:
    print(f"Error: {e}")

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
    <div style="background-color:blue;padding:10px">
    <h2 style="color:white;text-align:center;">Bankruptcy Detector</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    industrial_risk = st.text_input("industrial_risk")
    management_risk = st.text_input("management_risk")
    financial_flexibility = st.text_input("financial_flexibility")
    credibility = st.text_input("credibility")
    competitiveness = st.text_input("competitiveness")
    operating_risk = st.text_input("operating_risk")
    result=""
    if st.button("Predict"):
        result=predict_bankruptcy(industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets LEarn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
