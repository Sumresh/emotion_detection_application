import streamlit as st
from transformers import pipeline
import requests
import pandas as pd
from deep_translator import GoogleTranslator
from summarize import from_cohere
# from streamlit_lottie import st_lottie

def translate( text: str) -> str:
        return GoogleTranslator(source='auto', target='en').translate(text)

# Set the title of the main page
st.title("Text Emotion Detector")

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

if prompt :=st.chat_input("What is up?", key="user_input1"):

    # lottie_json = {"v":"5.8.1","fr":29.9700012207031,"ip":0,"op":41.0000016699642,"w":200,"h":200,"nm":"RDR","ddd":0,"assets":[],"layers":[{"ddd":0,"ind":1,"ty":4,"nm":"Shape Layer 1","sr":1,"ks":{"o":{"a":1,"k":[{"i":{"x":[0.833],"y":[0.833]},"o":{"x":[0.167],"y":[0.167]},"t":0,"s":[100]},{"t":40.0000016292334,"s":[80]}],"ix":11},"r":{"a":0,"k":0,"ix":10},"p":{"a":0,"k":[98,93.5,0],"ix":2,"l":2},"a":{"a":0,"k":[6,10.5,0],"ix":1,"l":2},"s":{"a":1,"k":[{"i":{"x":[0.833,0.833,0.833],"y":[0.833,0.833,0.833]},"o":{"x":[0.167,0.167,0.167],"y":[0.167,0.167,0.167]},"t":0,"s":[100,100,100]},{"t":40.0000016292334,"s":[150,150,100]}],"ix":6,"l":2}},"ao":0,"shapes":[{"ty":"gr","it":[{"ind":0,"ty":"sh","ix":1,"ks":{"a":0,"k":{"i":[[11.598,0],[0,-11.598],[-11.598,0],[0,11.598]],"o":[[-11.598,0],[0,11.598],[11.598,0],[0,-11.598]],"v":[[0,-21],[-21,0],[0,21],[21,0]],"c":true},"ix":2},"nm":"Path 1","mn":"ADBE Vector Shape - Group","hd":false},{"ty":"fl","c":{"a":0,"k":[0.898039275525,0.270588235294,0.145098039216,1],"ix":4},"o":{"a":0,"k":100,"ix":5},"r":1,"bm":0,"nm":"Fill 1","mn":"ADBE Vector Graphic - Fill","hd":false},{"ty":"tr","p":{"a":0,"k":[6,10.5],"ix":2},"a":{"a":0,"k":[0,0],"ix":1},"s":{"a":0,"k":[100,100],"ix":3},"r":{"a":0,"k":0,"ix":6},"o":{"a":0,"k":100,"ix":7},"sk":{"a":0,"k":0,"ix":4},"sa":{"a":0,"k":0,"ix":5},"nm":"Transform"}],"nm":"Ellipse 1","np":3,"cix":2,"bm":0,"ix":1,"mn":"ADBE Vector Group","hd":false}],"ip":0,"op":41.0000016699642,"st":0,"bm":0},{"ddd":0,"ind":2,"ty":4,"nm":"Shape Layer 2","sr":1,"ks":{"o":{"a":1,"k":[{"i":{"x":[0.833],"y":[0.833]},"o":{"x":[0.167],"y":[0.167]},"t":0,"s":[80]},{"t":40.0000016292334,"s":[60]}],"ix":11},"r":{"a":0,"k":0,"ix":10},"p":{"a":0,"k":[98,93.5,0],"ix":2,"l":2},"a":{"a":0,"k":[6,10.5,0],"ix":1,"l":2},"s":{"a":1,"k":[{"i":{"x":[0.833,0.833,0.833],"y":[0.833,0.833,0.833]},"o":{"x":[0.167,0.167,0.167],"y":[0.167,0.167,0.167]},"t":0,"s":[150,150,100]},{"t":40.0000016292334,"s":[200,200,100]}],"ix":6,"l":2}},"ao":0,"shapes":[{"ty":"gr","it":[{"ind":0,"ty":"sh","ix":1,"ks":{"a":0,"k":{"i":[[11.598,0],[0,-11.598],[-11.598,0],[0,11.598]],"o":[[-11.598,0],[0,11.598],[11.598,0],[0,-11.598]],"v":[[0,-21],[-21,0],[0,21],[21,0]],"c":true},"ix":2},"nm":"Path 1","mn":"ADBE Vector Shape - Group","hd":false},{"ty":"fl","c":{"a":0,"k":[0.898039275525,0.270588235294,0.145098039216,1],"ix":4},"o":{"a":0,"k":100,"ix":5},"r":1,"bm":0,"nm":"Fill 1","mn":"ADBE Vector Graphic - Fill","hd":false},{"ty":"tr","p":{"a":0,"k":[6,10.5],"ix":2},"a":{"a":0,"k":[0,0],"ix":1},"s":{"a":0,"k":[100,100],"ix":3},"r":{"a":0,"k":0,"ix":6},"o":{"a":0,"k":100,"ix":7},"sk":{"a":0,"k":0,"ix":4},"sa":{"a":0,"k":0,"ix":5},"nm":"Transform"}],"nm":"Ellipse 1","np":3,"cix":2,"bm":0,"ix":1,"mn":"ADBE Vector Group","hd":false}],"ip":0,"op":41.0000016699642,"st":0,"bm":0},{"ddd":0,"ind":3,"ty":4,"nm":"Shape Layer 3","sr":1,"ks":{"o":{"a":1,"k":[{"i":{"x":[0.833],"y":[0.833]},"o":{"x":[0.167],"y":[0.167]},"t":0,"s":[60]},{"t":40.0000016292334,"s":[50]}],"ix":11},"r":{"a":0,"k":0,"ix":10},"p":{"a":0,"k":[98,93.5,0],"ix":2,"l":2},"a":{"a":0,"k":[6,10.5,0],"ix":1,"l":2},"s":{"a":1,"k":[{"i":{"x":[0.833,0.833,0.833],"y":[0.833,0.833,0.833]},"o":{"x":[0.167,0.167,0.167],"y":[0.167,0.167,0.167]},"t":0,"s":[200,200,100]},{"t":40.0000016292334,"s":[250,250,100]}],"ix":6,"l":2}},"ao":0,"shapes":[{"ty":"gr","it":[{"ind":0,"ty":"sh","ix":1,"ks":{"a":0,"k":{"i":[[11.598,0],[0,-11.598],[-11.598,0],[0,11.598]],"o":[[-11.598,0],[0,11.598],[11.598,0],[0,-11.598]],"v":[[0,-21],[-21,0],[0,21],[21,0]],"c":true},"ix":2},"nm":"Path 1","mn":"ADBE Vector Shape - Group","hd":false},{"ty":"fl","c":{"a":0,"k":[0.898039275525,0.270588235294,0.145098039216,1],"ix":4},"o":{"a":0,"k":100,"ix":5},"r":1,"bm":0,"nm":"Fill 1","mn":"ADBE Vector Graphic - Fill","hd":false},{"ty":"tr","p":{"a":0,"k":[6,10.5],"ix":2},"a":{"a":0,"k":[0,0],"ix":1},"s":{"a":0,"k":[100,100],"ix":3},"r":{"a":0,"k":0,"ix":6},"o":{"a":0,"k":100,"ix":7},"sk":{"a":0,"k":0,"ix":4},"sa":{"a":0,"k":0,"ix":5},"nm":"Transform"}],"nm":"Ellipse 1","np":3,"cix":2,"bm":0,"ix":1,"mn":"ADBE Vector Group","hd":false}],"ip":0,"op":41.0000016699642,"st":0,"bm":0},{"ddd":0,"ind":4,"ty":4,"nm":"Shape Layer 4","sr":1,"ks":{"o":{"a":1,"k":[{"i":{"x":[0.833],"y":[0.833]},"o":{"x":[0.167],"y":[0.167]},"t":0,"s":[40]},{"t":40.0000016292334,"s":[20]}],"ix":11},"r":{"a":0,"k":0,"ix":10},"p":{"a":0,"k":[98,93.5,0],"ix":2,"l":2},"a":{"a":0,"k":[6,10.5,0],"ix":1,"l":2},"s":{"a":1,"k":[{"i":{"x":[0.833,0.833,0.833],"y":[0.833,0.833,0.833]},"o":{"x":[0.167,0.167,0.167],"y":[0.167,0.167,0.167]},"t":0,"s":[250,250,100]},{"t":40.0000016292334,"s":[300,300,100]}],"ix":6,"l":2}},"ao":0,"shapes":[{"ty":"gr","it":[{"ind":0,"ty":"sh","ix":1,"ks":{"a":0,"k":{"i":[[11.598,0],[0,-11.598],[-11.598,0],[0,11.598]],"o":[[-11.598,0],[0,11.598],[11.598,0],[0,-11.598]],"v":[[0,-21],[-21,0],[0,21],[21,0]],"c":true},"ix":2},"nm":"Path 1","mn":"ADBE Vector Shape - Group","hd":false},{"ty":"fl","c":{"a":0,"k":[0.898039275525,0.270588235294,0.145098039216,1],"ix":4},"o":{"a":0,"k":100,"ix":5},"r":1,"bm":0,"nm":"Fill 1","mn":"ADBE Vector Graphic - Fill","hd":false},{"ty":"tr","p":{"a":0,"k":[6,10.5],"ix":2},"a":{"a":0,"k":[0,0],"ix":1},"s":{"a":0,"k":[100,100],"ix":3},"r":{"a":0,"k":0,"ix":6},"o":{"a":0,"k":100,"ix":7},"sk":{"a":0,"k":0,"ix":4},"sa":{"a":0,"k":0,"ix":5},"nm":"Transform"}],"nm":"Ellipse 1","np":3,"cix":2,"bm":0,"ix":1,"mn":"ADBE Vector Group","hd":false}],"ip":0,"op":41.0000016699642,"st":0,"bm":0},{"ddd":0,"ind":5,"ty":4,"nm":"Shape Layer 5","sr":1,"ks":{"o":{"a":1,"k":[{"i":{"x":[0.833],"y":[0.833]},"o":{"x":[0.167],"y":[0.167]},"t":0,"s":[20]},{"t":40.0000016292334,"s":[0]}],"ix":11},"r":{"a":0,"k":0,"ix":10},"p":{"a":0,"k":[98,93.5,0],"ix":2,"l":2},"a":{"a":0,"k":[6,10.5,0],"ix":1,"l":2},"s":{"a":1,"k":[{"i":{"x":[0.833,0.833,0.833],"y":[0.833,0.833,0.833]},"o":{"x":[0.167,0.167,0.167],"y":[0.167,0.167,0.167]},"t":0,"s":[300,300,100]},{"t":40.0000016292334,"s":[350,350,100]}],"ix":6,"l":2}},"ao":0,"shapes":[{"ty":"gr","it":[{"ind":0,"ty":"sh","ix":1,"ks":{"a":0,"k":{"i":[[11.598,0],[0,-11.598],[-11.598,0],[0,11.598]],"o":[[-11.598,0],[0,11.598],[11.598,0],[0,-11.598]],"v":[[0,-21],[-21,0],[0,21],[21,0]],"c":true},"ix":2},"nm":"Path 1","mn":"ADBE Vector Shape - Group","hd":false},{"ty":"fl","c":{"a":0,"k":[0.898039275525,0.270588235294,0.145098039216,1],"ix":4},"o":{"a":0,"k":100,"ix":5},"r":1,"bm":0,"nm":"Fill 1","mn":"ADBE Vector Graphic - Fill","hd":false},{"ty":"tr","p":{"a":0,"k":[6,10.5],"ix":2},"a":{"a":0,"k":[0,0],"ix":1},"s":{"a":0,"k":[100,100],"ix":3},"r":{"a":0,"k":0,"ix":6},"o":{"a":0,"k":100,"ix":7},"sk":{"a":0,"k":0,"ix":4},"sa":{"a":0,"k":0,"ix":5},"nm":"Transform"}],"nm":"Ellipse 1","np":3,"cix":2,"bm":0,"ix":1,"mn":"ADBE Vector Group","hd":false}],"ip":0,"op":41.0000016699642,"st":0,"bm":0},{"ddd":0,"ind":6,"ty":4,"nm":"Shape Layer 6","sr":1,"ks":{"o":{"a":0,"k":0,"ix":11},"r":{"a":0,"k":0,"ix":10},"p":{"a":0,"k":[98,93.5,0],"ix":2,"l":2},"a":{"a":0,"k":[6,10.5,0],"ix":1,"l":2},"s":{"a":0,"k":[350,350,100],"ix":6,"l":2}},"ao":0,"shapes":[{"ty":"gr","it":[{"ind":0,"ty":"sh","ix":1,"ks":{"a":0,"k":{"i":[[11.598,0],[0,-11.598],[-11.598,0],[0,11.598]],"o":[[-11.598,0],[0,11.598],[11.598,0],[0,-11.598]],"v":[[0,-21],[-21,0],[0,21],[21,0]],"c":true},"ix":2},"nm":"Path 1","mn":"ADBE Vector Shape - Group","hd":false},{"ty":"fl","c":{"a":0,"k":[0.898039275525,0.270588235294,0.145098039216,1],"ix":4},"o":{"a":0,"k":100,"ix":5},"r":1,"bm":0,"nm":"Fill 1","mn":"ADBE Vector Graphic - Fill","hd":false},{"ty":"tr","p":{"a":0,"k":[6,10.5],"ix":2},"a":{"a":0,"k":[0,0],"ix":1},"s":{"a":0,"k":[100,100],"ix":3},"r":{"a":0,"k":0,"ix":6},"o":{"a":0,"k":100,"ix":7},"sk":{"a":0,"k":0,"ix":4},"sa":{"a":0,"k":0,"ix":5},"nm":"Transform"}],"nm":"Ellipse 1","np":3,"cix":2,"bm":0,"ix":1,"mn":"ADBE Vector Group","hd":false}],"ip":0,"op":41.0000016699642,"st":0,"bm":0}],"markers":[]}
    # # Append user prompt to the session state
    # st_lottie(lottie_json, height=200, key="loader")
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