import streamlit as st
import pandas as pd
import os
from whatstk import df_from_txt_whatsapp
from transformers import pipeline

# Load the WhatsApp chat data
# df = df_from_txt_whatsapp("C:\Sumresh N\Aiml\emotion_detection_application\pages\chat.txt")

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

# Define Streamlit app layout
def main():
    st.title("WhatsApp 🗨️ Chat Analyzer")

    # File uploader to upload chat data
    file = st.file_uploader("Upload File", type=['txt'])
    if file is not None:
        save_directory = os.getcwd()
        os.makedirs(save_directory, exist_ok=True)
        filename = file.name
        save_path = os.path.join(save_directory, "audio.wav")
        with open(save_path, "wb") as f:
            f.write(file.getvalue())
        st.success(f"File saved successfully")
    
    df=df_from_txt_whatsapp("file.txt")
    # Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Date inputs for filtering
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    # Convert start_date and end_date to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter DataFrame for dates within the specified range
    filtered_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    filtered_df = filtered_df[~filtered_df['message'].str.contains("<Media omitted>")]
    # Display filtered DataFrame
    if st.button("Get Emotion"):
        with st.spinner('Analyzing...'):
            messages = filtered_df['message'].tolist()
            emotions = []
            for i in messages:
                if len(i) > 20:
                    model_res = classifier(i[:20])
                else:
                    model_res = classifier(i)
                emotions.append(model_res[0][0]['label'])

            filtered_df["Results"] = emotions

            # Remove rows containing '<Media omitted>' in the "Results" column

            pd.set_option('display.max_columns', None)  # Show all columns
            pd.set_option('display.expand_frame_repr', False)  # Prevent DataFrame from being truncated

            # Display the DataFrame with adjusted column width
            st.write(filtered_df)


# Run the Streamlit app
if __name__ == "__main__":
    main()
