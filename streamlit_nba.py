import streamlit as st
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import requests
import numpy as np

### Config
st.set_page_config(
    page_title="Invest on this NBA player",
    page_icon="💸",
    layout="wide"
)


# api url 

api_url = 'http://localhost:4000'

### App
st.title("Is it worth investing in this NBA player ?")

st.markdown("""
    Welcome to this user interface
    Find out which NBA player you should invest in !
""")


st.markdown("---")

## Raw data 
st.subheader('Raw data')
st.write("This gives you an overview of a dataset composed of players with their NBA statistics")

response = requests.get(api_url + '/preview?rows=10')

if response.status_code == 200:
    data = response.json()
    print(data)
    data = eval(data)
    df = pd.DataFrame(data)
    print(df)
    print(df.info())
    st.dataframe(df)
else:
    st.write('API Error:', response.text)



## Player statistics
st.subheader('Player statistics')
st.write("Enter player statistics to predict their future in the NBA.")

PLAYER = st.text_input("Player name:", "")
GP = st.number_input("Games played", min_value = 0, max_value = 82)
MIN = st.number_input("Minutes played", min_value =0, max_value = 78)
PTS = st.number_input("Points scored", min_value = 0)
FGA = st.number_input("Field Goals Attemped", min_value = 0)
FGM = st.number_input("Field Goals Made", min_value = 0,max_value=FGA)
P3A = st.number_input("3P Attemped", min_value = 0)
P3M = st.number_input("3P Made", min_value = 0, max_value=P3A)
FTA = st.number_input("Field Throw Attemped", min_value = 0)
FTM = st.number_input("Field Throw Made", min_value = 0, max_value=FTA)
OREB = st.number_input("Offensive rebounds", min_value = 0)
DREB = st.number_input("Deffensive rebounds", min_value = 0)
AST = st.number_input("Assists", min_value = 0)
STL = st.number_input("Steals", min_value = 0)
BLK = st.number_input("Blocks", min_value = 0)
TOV = st.number_input("Turnovers", min_value = 0)


if st.button("Predict"):
    try:
        input_data = {
        'GP': GP, 'MIN': MIN, 'PTS': PTS, 'FGM': FGM, 'FGA': FGA, 
        'P3_Made': P3M, 'P3A': P3A, 'FTM': FTM, 'FTA': FTA, 'OREB': OREB,
        'DREB': DREB, 'AST': AST, 'STL': STL, 'BLK': BLK, 'TOV': TOV
        }
        
        response = requests.post(api_url + '/predict', json= input_data)
        if response.status_code == 200:
            response_json = response.json()
            prediction = response_json["prediction"]
            #print(prediction)
            if prediction == 1:
                st.markdown(f"<div style='font-size: 24px; color: #0066cc;'>It's worth investing in {PLAYER}</div>", unsafe_allow_html=True)
            else:
                 st.markdown(f"<div style='font-size: 24px; color: #FF0000;'>It's NOT worth investing in {PLAYER}</div>", unsafe_allow_html=True)
        else:
            st.error("Failed to get a prediction from the API. Please try again.")
            
    except Exception as e:
        st.error(f"An error occurred while making the prediction: {e}")


### Side bar 
st.sidebar.header("Check pages")
st.sidebar.markdown("""
    * [Raw data](#Raw-data)
    * [Player statistics](#Prediction)
""")
e = st.sidebar.empty()
e.write("")
st.sidebar.write("Made by [Kenneth]")



### Footer 
empty_space, footer = st.columns([1, 2])

with empty_space:
    st.write("")

with footer:
    st.markdown("""
        🍇
        If you want to learn more, contact [Xxxxx](https://..../) 📖
    """)