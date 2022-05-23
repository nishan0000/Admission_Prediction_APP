import pickle
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler
import numpy as np

# loading in the model to predict on the data
filename = 'Admission_LinearRegression2.sav'
model = joblib.load(open(filename, 'rb'))


def welcome():
    return 'welcome all'


# defining the function which will make the prediction using
# the data which the user inputs
def prediction(GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA, Research):
    scaler = StandardScaler()
    prediction = model.predict([[GRE_Score, TOEFL_Score, University_Rating, SOP, LOR, CGPA, Research]])
    print(prediction)
    return prediction


# this is the main function in which we define our webpage
def main():
    # giving the webpage a title
    st.title("Admission Probability Prediction")

    # the following lines create text boxes in which the user can enter
    # the data required to make the prediction
    GRE_Score = st.text_input("GRE_Score")
    TOEFL_Score = st.text_input("TOEFL_Score")
    University_Rating = st.text_input("University_Rating")
    SOP = st.text_input("SOP")
    LOR = st.text_input("LOR")
    CGPA = st.text_input("CGPA")
    Research = st.text_input("Research")
    result = ""

    # the below line ensures that when the button called 'Predict' is clicked,
    # the prediction function defined above is called to make the prediction
    # and store it in the variable result
    if st.button("Predict"):
        result = prediction(float(GRE_Score), float(TOEFL_Score), float(University_Rating), float(SOP), float(LOR),
                            float(CGPA), float(Research))
    st.success('The Percentage chance to get admitted is {}%'.format(np.round(result[0]*100,2)))


if __name__ == '__main__':
    main()
