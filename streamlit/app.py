from logging import exception

import streamlit as st
import os
from openai import OpenAI
from pyexpat.errors import messages

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key = "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",
)

def get_openai_responce(user_input):
    try:
        response = client.chat.completions.create(
            model = "deepseek/deepseek-r1:free",
            messages =[
                {"role": "system", "content": "You are a struggling artists who's scared his job is going to be taken by AI"},
                {"role": "user", "content": user_input},
            ]
        )

        if response.choices:
            return response.choices[0].message.content.strip()
        else:
            return "No response"
    except Exception as e:
        return f"An error occurred: {e}"

st.title("Chatbot")
user_input = st.text_input("Ask away")
if st.button("Submit"):
    if(user_input):
        chatbot_response = get_openai_responce(user_input)
        st.write(f"Chatbot: {chatbot_response}")