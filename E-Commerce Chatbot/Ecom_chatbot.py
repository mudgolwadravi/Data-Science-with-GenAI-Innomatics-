import os
import streamlit as st 
from dotenv import load_dotenv

from google import genai

load_dotenv('.api_key')

os.environ['GEMINI_API_KEY']=os.getenv('API_KEY')

if 'client' not in st.session_state:
    st.session_state.client=genai.Client()

client=st.session_state.client

system_prompt = """You are a smart e-commerce assistant. 
Provide concise product details (features, specs, pros/cons) in 4–5 lines. 
Compare current prices with past trends, highlight discounts or cashback, and advise whether to buy now or wait. 
Keep responses short, clear, and user-friendly, like a modern shopping site."""




st.title("E-Commerce Chatbot")
st.write("Type your message below to chat with the model.")

if 'chat_session' not in st.session_state:
    st.session_state.chat_session=client.chats.create(
            model='gemini-2.5-flash',
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
            )
    )

if 'messages' not in st.session_state:
    st.session_state.messages=[]
print(st.session_state)

# for role,text in st.session_state.messages:
#     if role=='user':
#         st.markdown(f"**You:**{text}")
#     else:
#         st.markdown(f"**Bot:**{text}")

for role, text in st.session_state.messages:
    if role == "user":
        with st.chat_message("user"):
            st.write(text)
    else:
        with st.chat_message("assistant"):
            st.write(text)


user_input=st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append(('user',user_input))

    chat=st.session_state.chat_session
    response=chat.send_message(user_input)

    bot_reply=response.text
    st.session_state.messages.append(('bot',bot_reply))

    st.rerun()