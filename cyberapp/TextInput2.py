import streamlit as st
from transformers import pipeline
import requests
import pandas as pd
from deep_translator import GoogleTranslator
from summarize import from_cohere
from streamlit_lottie import st_lottie
import json
import os

def translate( text: str) -> str:
        return GoogleTranslator(source='auto', target='en').translate(text)

# Set the title of the main page
st.title("Text Emotion Detector")

# Function to display messages with reversed roles
def display_message(role, content):
    with st.chat_message(role):
        st.markdown(content)

def lottie_anim():
     relative_path = "loading.json"
     absolute_path = os.path.abspath(relative_path)
     with open(absolute_path, "r") as f:
          return json.load(f)

# Check if "messages" exist in the session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear previous prompts and keep only the latest one
st.session_state.messages = []
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
# Chat input for user prompt

if prompt :=st.chat_input("What is up?", key="user_input1"):
    st_lottie(lottie_anim(), speed=1, reverse=False, loop=True, quality="medium", height=None, width=None, key=None)
    
    
    # # Append user prompt to the session state
    
    # Display user prompt
    # display_message("user", prompt)  

    # translator = Translator() 
    output=translate(prompt)
    translated_meaning=prompt + " \n " + output
    translated_meaning= f"Input:  \n{prompt}  \nTranslated to English:  \n{output}"
    display_message("user", translated_meaning)
    model_outputs = classifier(output)
 
    
    result = model_outputs[0][0]['label'] + " : " + str(round(model_outputs[0][0]['score'] * 100, 2)) + "%"
    result1 = model_outputs[0][1]['label'] + " : " + str(round(model_outputs[0][1]['score'] * 100, 2)) + "%"
    result2 = model_outputs[0][2]['label'] + " : " + str(round(model_outputs[0][2]['score'] * 100, 2)) + "%"


    # display_message("assistant", result)
    display_message("assistant", f"- {result}  \n- {result1}  \n- {result2}")

    data=model_outputs[0]

    df = pd.DataFrame(data)



    df_sorted = df.sort_values(by='score', ascending=False)

    st.bar_chart(df.set_index('label')['score'], use_container_width=True, color="#87CEEB")
    display_message('assistant', f"Exlpaination:  \n{from_cohere(output, [result, result1, result2])}")

if render_animation:
    