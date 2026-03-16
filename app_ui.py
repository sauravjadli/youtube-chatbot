import streamlit as st
from app import build_chain

st.title("YouTube Chatbot 🎥")
url = st.text_input("Enter YouTube URL")

if url:
    if "chain" not in st.session_state:
        with st.spinner("Loading video..."):
            st.session_state.chain = build_chain(url)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    prompt = st.chat_input("Ask something about the video")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        response = st.session_state.chain.invoke(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()