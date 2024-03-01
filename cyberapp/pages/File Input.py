import streamlit as st
from transformers import pipeline
import pandas as pd
from docx import Document
import pdfplumber
from deep_translator import GoogleTranslator

# Function to translate text
def translate(text: str) -> str:
    return GoogleTranslator(source='auto', target='en').translate(text)

# Set the title of the main page
st.title("Upload a File to Extract Emotion")

# Function to display messages with reversed roles
def display_message(role, content):
    with st.chat_message(role):
        st.markdown(content)

# Check if "messages" exist in the session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Clear previous prompts and keep only the latest one
st.session_state.messages = []

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

# Chat input for user prompt
file = st.file_uploader("Upload File", type=['txt', 'docx', 'pdf'])

if file:
    # Read text from the uploaded file based on file type
    if file.type == 'text/plain':  # For .txt files
        text = file.getvalue().decode("utf-8")
    elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':  # For .docx files
        doc = Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    elif file.type == 'application/pdf':  # For .pdf files
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"

    # Translate the text
    translated_text = translate(text)

    # Run the emotion detection model on the translated text
    model_outputs = classifier(translated_text)

    # Extract emotion labels and scores
    result = model_outputs[0][0]['label'] + " : " + str(round(model_outputs[0][0]['score'] * 100, 2)) + "%"

    # Display emotion result
    display_message("assistant", result)

    # Create DataFrame from model outputs for visualization
    data = model_outputs[0]
    df = pd.DataFrame(data)
    df_sorted = df.sort_values(by='score', ascending=False)

    # Display bar chart of emotion scores
    st.bar_chart(df.set_index('label')['score'], use_container_width=True, color="#87CEEB")
