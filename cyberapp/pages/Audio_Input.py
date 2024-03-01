import streamlit as st
import os
from deep_translator import GoogleTranslator
from audio_to_text import audio_to_text
from audio_recorder_streamlit import audio_recorder
from transformers import pipeline
import pandas as pd
from streamlit_extras import row
from summarize import from_cohere
from streamlit_extras.add_vertical_space import add_vertical_space



def display_message(role, content):
  with st.chat_message(role):
    st.markdown(content)

def start_model_file():
  
  message = audio_to_text()
  translated_text = GoogleTranslator(source='auto', target='en').translate(message['Transcription'])

  display_message("user", f"Your Input: {message['Transcription']}  \nTranslated to ENGLISH: {translated_text}")

  classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
  model_outputs = classifier(translated_text)
  
  result = model_outputs[0][0]['label'] + " : " + str(round(model_outputs[0][0]['score'] * 100, 2)) + "%"
  result1 = model_outputs[0][1]['label'] + " : " + str(round(model_outputs[0][1]['score'] * 100, 2)) + "%"
  result2 = model_outputs[0][2]['label'] + " : " + str(round(model_outputs[0][2]['score'] * 100, 2)) + "%"
  
  display_message("assistant", f" - {result}  \n- {result1}  \n- {result2}")

  data = model_outputs[0]
  df = pd.DataFrame(data)
  st.bar_chart(df.set_index('label')['score'], use_container_width=True, color="#87CEEB")
  display_message('assistant', f"Exlpaination:  \n{from_cohere(translated_text, [result, result1, result2])}")

def start_model_live():
 
  message = audio_to_text()
  translated_text = GoogleTranslator(source='auto', target='en').translate(message['Transcription'])
  display_message("assistant", f"Your Input: {message['Transcription']}  \nTranslated to ENGLISH: {translated_text}")

  classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
  model_outputs = classifier(translated_text)
  # result = model_outputs[0][0]['label'] + " : " + str(round(model_outputs[0][0]['score'] * 100, 2)) + "%"

  # display_message("assistant", result)
  result = model_outputs[0][0]['label'] + " : " + str(round(model_outputs[0][0]['score'] * 100, 2)) + "%"
  result1 = model_outputs[0][1]['label'] + " : " + str(round(model_outputs[0][1]['score'] * 100, 2)) + "%"
  result2 = model_outputs[0][2]['label'] + " : " + str(round(model_outputs[0][2]['score'] * 100, 2)) + "%"
  
  display_message("assistant", f" - {result}  \n- {result1}  \n- {result2}")

  data = model_outputs[0]
#   df = pd.DataFrame(data)
#   # st.bar_chart(df.set_index('label')['score'], use_container_width=True, color="#87CEEB")

#   df['score_normalized'] = df['score'] * 100

# # Plot the normalized scores as a bar chart
#   st.bar_chart(df.set_index('label')['score_normalized'], use_container_width=True, color="#87CEEB")
  df = pd.DataFrame(data)
  st.bar_chart(df.set_index('label')['score'], use_container_width=True, color="#87CEEB")
  display_message('assistant', f"Exlpaination:  \n{from_cohere(translated_text, [result, result1, result2])}")



st.title("Emotion 🎤  Detection")

choice = st.selectbox("Pick one", ["Upload_File", "Record Audio"])

    




if choice=="Upload_File" :
  uploaded_file = st.file_uploader("Choose a file", type=["csv", "txt", "wav", "mp3", "m4a","MPEG"])
  if uploaded_file is not None:
    save_directory = os.getcwd()
    os.makedirs(save_directory, exist_ok=True)
    filename = uploaded_file.name
    save_path = os.path.join(save_directory, "audio.wav")
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    st.success(f"File saved successfully")
    st.button('Run', on_click=start_model_file)



if choice=="Record Audio":
  add_vertical_space(3)
  
  audio_bytes = audio_recorder(text="Click to record: ", neutral_color="#F47174", recording_color="#6FC276")
  if audio_bytes is not None:
    voice = st.audio(audio_bytes, format="audio/wav")
    save_directory = os.getcwd()
    os.makedirs(save_directory, exist_ok=True)
    filename = "audio.wav"
    save_path = os.path.join(save_directory, filename)
    with open(save_path, "wb") as f:
        f.write(audio_bytes)
    st.success(f"File saved successfully at {save_path}")
    st.button('Run', on_click=start_model_live)

