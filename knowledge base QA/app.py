import os
import time
import streamlit as st
from dotenv import load_dotenv
from utils.bedrock import retrieve_from_kb
from utils.gemini import ask_gemini

load_dotenv()

st.set_page_config(page_title="KB + Gemini Chat", page_icon="💬")
st.title("Bedrock KB + Gemini 2.5 Flash")

if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0
if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Enter your question")

if st.button("Ask"):
    if not os.getenv("KNOWLEDGE_BASE_ID"):
        st.error("KNOWLEDGE_BASE_ID is missing from .env")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        elapsed = time.time() - st.session_state.last_request_time
        if elapsed < 15:
            st.warning(f"Please wait {int(15 - elapsed)}s before asking again.")
        else:
            try:
                with st.spinner("Retrieving from Knowledge Base..."):
                    context, bedrock_answer = retrieve_from_kb(question)

                # Use Gemini to refine the answer using the retrieved context
                with st.spinner("Refining answer with Gemini..."):
                    gemini_answer = ask_gemini(context, question)

                st.session_state.last_request_time = time.time()
                st.session_state.history.append({
                    "question": question,
                    "gemini_answer": gemini_answer,
                    "bedrock_answer": bedrock_answer,
                    "context": context
                })

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("If throttled, wait 1-2 minutes and try again.")

for item in reversed(st.session_state.history):
    st.markdown(f"**Q:** {item['question']}")
    st.subheader("Gemini Answer")
    st.write(item["gemini_answer"])
    with st.expander("Bedrock Titan Answer (for comparison)"):
        st.write(item["bedrock_answer"])
    with st.expander("Retrieved Context"):
        st.write(item["context"] if item["context"] else "No context returned.")
    st.divider()